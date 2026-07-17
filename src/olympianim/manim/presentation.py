"""Safety checks for generated code and isolated Manim rendering."""

from __future__ import annotations

import ast
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from collections.abc import Callable, Collection, Mapping
from pathlib import Path, PurePosixPath

from olympianim.config import DEFAULT_RENDER_QUALITY, DEFAULT_RENDER_TIMEOUT_SECONDS
from olympianim.manim.usage_events import USAGE_PATH_ENV, read_usage_events
from olympianim.schemas.render import ManimCodeResult, RenderResult, VideoMode
from olympianim.utils.logging import redact

_FORBIDDEN_CALLS = {
    "__import__",
    "compile",
    "delattr",
    "eval",
    "exec",
    "getattr",
    "globals",
    "input",
    "locals",
    "open",
    "setattr",
    "vars",
}
_ALLOWED_IMPORT_ROOTS = {"__future__", "manim", "math", "numpy", "typing"}
_CONFIGURED_VOICEOVER_IMPORT = "from olympianim.manim.voiceover import ConfiguredVoiceoverScene"
_VOICEOVER_MODULE = "olympianim.manim.voiceover"
_WATERMARK_MIXIN = "OlympianimSourceWatermarkMixin"
_WATERMARK_MARKER = "# olympianim:source-watermark"
_QUALITY_FLAGS = {
    "low_quality": "-ql",
    "medium_quality": "-qm",
    "high_quality": "-qh",
    "production_quality": "-qk",
}
_RENDER_ENV_ALLOWLIST = frozenset(
    {
        "PATH",
        "HOME",
        "USER",
        "LANG",
        "LANGUAGE",
        "LC_ALL",
        "LC_CTYPE",
        "TMPDIR",
        "TEMP",
        "TMP",
        "XDG_CACHE_HOME",
        "XDG_CONFIG_HOME",
        "FONTCONFIG_FILE",
        "FONTCONFIG_PATH",
        "LD_LIBRARY_PATH",
        "DYLD_LIBRARY_PATH",
        "SSL_CERT_FILE",
        "SSL_CERT_DIR",
        "REQUESTS_CA_BUNDLE",
    }
)


class PresentationSafetyError(ValueError):
    """Raised when app-owned rendering safeguards cannot be applied."""


def check_generated_code_safety(
    code: str,
    *,
    require_voiceover: bool,
    allowed_image_paths: Collection[str] = (),
) -> list[str]:
    """Return only execution-safety and app-integration violations.

    Manim remains responsible for deciding whether its API is used correctly.
    This check intentionally makes no judgment about layout, visual quality,
    grouping choices, text classes, or the mathematical content on screen.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return [f"Código Manim inválido: {exc.msg}."]

    errors: list[str] = []
    has_scene = False
    has_configured_voiceover_scene = False
    has_construct = False
    has_voiceover_block = False
    has_direct_voice_configuration = False
    imports_configured_voiceover_scene = False
    imports_direct_voiceover = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports_direct_voiceover = imports_direct_voiceover or alias.name.startswith(
                    "manim_voiceover"
                )
                if alias.name.split(".", 1)[0] not in _ALLOWED_IMPORT_ROOTS:
                    errors.append(f"Importação não permitida: {alias.name}.")
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".", 1)[0]
            imports_direct_voiceover = imports_direct_voiceover or root == "manim_voiceover"
            imports_configured_voiceover_scene = imports_configured_voiceover_scene or (
                node.module == "olympianim.manim.voiceover"
                and any(alias.name == "ConfiguredVoiceoverScene" for alias in node.names)
            )
            is_voiceover_import = node.module == _VOICEOVER_MODULE and all(
                alias.name == "ConfiguredVoiceoverScene" for alias in node.names
            )
            if root not in _ALLOWED_IMPORT_ROOTS and not is_voiceover_import:
                errors.append(f"Importação não permitida: {node.module or ''}.")
        elif isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            errors.append(f"Acesso interno não permitido: {node.attr}.")
        elif isinstance(node, ast.ClassDef):
            has_scene = has_scene or any(_node_name(base).endswith("Scene") for base in node.bases)
            has_configured_voiceover_scene = has_configured_voiceover_scene or any(
                _node_name(base) == "ConfiguredVoiceoverScene" for base in node.bases
            )
            has_construct = has_construct or any(
                isinstance(child, ast.FunctionDef) and child.name == "construct"
                for child in node.body
            )
        elif isinstance(node, ast.Call):
            call_name = _node_name(node.func)
            if call_name in _FORBIDDEN_CALLS:
                errors.append(f"Chamada não permitida no código gerado: {call_name}.")
            has_voiceover_block = has_voiceover_block or call_name == "voiceover"
            has_direct_voice_configuration = has_direct_voice_configuration or call_name in {
                "set_speech_service",
                "init_voiceover",
            }
            if call_name == "ImageMobject":
                errors.extend(_validate_image_mobject_call(node, allowed_image_paths))
    if not has_scene:
        errors.append("O código deve definir uma cena Manim válida.")
    if not has_construct:
        errors.append("A cena deve definir construct().")
    if require_voiceover:
        if not has_configured_voiceover_scene:
            errors.append(
                "A cena deve herdar de ConfiguredVoiceoverScene quando a narração estiver ativa."
            )
        if not imports_configured_voiceover_scene:
            errors.append(
                "O código deve importar ConfiguredVoiceoverScene da camada de voz do aplicativo."
            )
        if not has_voiceover_block:
            errors.append(
                "O código deve usar blocos self.voiceover(...) quando a narração estiver ativa."
            )
        if imports_direct_voiceover:
            errors.append(
                "O código não deve importar manim_voiceover diretamente; o aplicativo controla o provedor."
            )
        if has_direct_voice_configuration:
            errors.append(
                "O código não deve configurar serviços de voz; o aplicativo usa a escolha do usuário."
            )
    elif (
        has_configured_voiceover_scene
        or has_voiceover_block
        or imports_configured_voiceover_scene
        or imports_direct_voiceover
    ):
        errors.append("O código não deve incluir narração quando ela não foi solicitada.")
    return errors


def _validate_image_mobject_call(
    node: ast.Call,
    allowed_image_paths: Collection[str],
) -> list[str]:
    if not node.args or not isinstance(node.args[0], ast.Constant):
        return ["ImageMobject deve usar diretamente um caminho de imagem autorizado."]
    value = node.args[0].value
    if not isinstance(value, str):
        return ["ImageMobject deve receber um caminho de imagem em texto."]
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts:
        return [f"Caminho de imagem não permitido: {value}."]
    if path.as_posix() not in allowed_image_paths:
        return [f"Imagem não registrada no projeto: {value}."]
    return []


def prepare_voiceover_code(code: str, *, require_voiceover: bool) -> str:
    """Inject the application-owned voice scene import into generated code.

    The builder writes ``ConfiguredVoiceoverScene`` when narration is enabled,
    but does not own provider imports.  The normalizer also repairs the common
    model-only alias ``VoiceoverScene`` without accepting speech-service setup.
    Validation remains responsible for rejecting any service configuration.
    """
    if not require_voiceover:
        return code
    normalized = re.sub(
        r"^\s*from\s+manim_voiceover(?:\.[A-Za-z0-9_.]+)?\s+import\s+[^\n]+\n?",
        "",
        code,
        flags=re.MULTILINE,
    )
    normalized = re.sub(
        r"^\s*import\s+manim_voiceover(?:\s+as\s+[A-Za-z_][A-Za-z0-9_]*)?\s*\n?",
        "",
        normalized,
        flags=re.MULTILINE,
    )
    normalized = re.sub(r"\bVoiceoverScene\b", "ConfiguredVoiceoverScene", normalized)
    if _CONFIGURED_VOICEOVER_IMPORT not in normalized:
        normalized = _CONFIGURED_VOICEOVER_IMPORT + "\n" + normalized.lstrip()
    return normalized


def prepare_source_watermark_code(code: str, source: str) -> str:
    """Return code with exactly one app-owned source watermark."""
    code = strip_source_watermark_code(code)
    label = source.strip()
    if not label:
        return code
    scene_name = presentation_scene_name(code, require_voiceover=False)
    class_pattern = re.compile(rf"(class\s+{re.escape(scene_name)}\s*\()")
    watermarked, replacements = class_pattern.subn(rf"\1{_WATERMARK_MIXIN}, ", code, count=1)
    if replacements != 1:
        raise PresentationSafetyError("Não foi possível aplicar a marca d'água à cena.")
    mixin = f"""{_WATERMARK_MARKER}
class {_WATERMARK_MIXIN}:
    def setup(self):
        super().setup()
        self._olympianim_source_watermark = (
            Text({f"Fonte: {label}"!r}, font_size=18, color=GRAY_B)
            .set_opacity(0.3)
            .to_edge(DOWN, buff=0.15)
            .set_z_index(-100)
        )
        super().add(self._olympianim_source_watermark)

    def clear(self):
        super().clear()
        super().add(self._olympianim_source_watermark)


"""
    future_match = re.match(r"((?:from __future__ import [^\n]+\n)+)", watermarked)
    if future_match is None:
        return mixin + watermarked.lstrip()
    insertion = future_match.end()
    return watermarked[:insertion] + "\n" + mixin + watermarked[insertion:].lstrip()


def strip_source_watermark_code(code: str) -> str:
    """Remove app-owned watermark definitions and inheritance from source code."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code

    lines = code.splitlines(keepends=True)
    class_ranges = [
        (node.lineno - 1, node.end_lineno or node.lineno)
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == _WATERMARK_MIXIN
    ]
    for start, end in reversed(class_ranges):
        del lines[start:end]
    stripped = "".join(lines)
    stripped = re.sub(
        rf"(?m)^\s*{re.escape(_WATERMARK_MARKER)}\s*\n?",
        "",
        stripped,
    )

    class_header = re.compile(r"(class\s+\w+\s*\()(?P<bases>.*?)(\)\s*:)", re.DOTALL)

    def remove_mixin_base(match: re.Match[str]) -> str:
        bases = [
            base.strip()
            for base in match.group("bases").split(",")
            if base.strip() and base.strip() != _WATERMARK_MIXIN
        ]
        return f"{match.group(1)}{', '.join(bases)}{match.group(3)}"

    stripped = class_header.sub(remove_mixin_base, stripped)
    return stripped.lstrip("\n")


def presentation_scene_name(code: str, *, require_voiceover: bool) -> str:
    """Return the expected Manim scene subclass name from validated code."""
    tree = ast.parse(code)
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        bases = {_node_name(base) for base in node.bases}
        if require_voiceover and "ConfiguredVoiceoverScene" in bases:
            return node.name
        if not require_voiceover and any(base.endswith("Scene") for base in bases):
            return node.name
    expected = "VoiceoverScene" if require_voiceover else "Scene"
    raise PresentationSafetyError(f"O código não define uma cena {expected} válida.")


RenderCommandRunner = Callable[..., subprocess.CompletedProcess[str]]


class PresentationRenderer:
    """Render generated scenes in an isolated Manim subprocess."""

    def __init__(
        self,
        *,
        cancellation_check: Callable[[], None] | None = None,
        timeout_seconds: int = DEFAULT_RENDER_TIMEOUT_SECONDS,
        command_runner: RenderCommandRunner | None = None,
    ) -> None:
        self.cancellation_check = cancellation_check or (lambda: None)
        self.timeout_seconds = timeout_seconds
        self.command_runner = command_runner or _run_cancellable_process

    def render(
        self,
        code: ManimCodeResult,
        *,
        project_directory: Path,
        mode: VideoMode = "presentation",
        api_key: str = "",
        voice_provider: str = "",
        voice_model: str = "",
        voice: str = "",
        voice_language: str = "",
        voice_speed: float = 1.0,
        voice_prompt_template: str = "{transcript}",
        voiceover_enabled: bool = False,
        quality: str = DEFAULT_RENDER_QUALITY,
    ) -> RenderResult:
        """Render ``code`` and capture complete subprocess diagnostics."""
        if quality not in _QUALITY_FLAGS:
            raise ValueError(f"Qualidade de renderização inválida: {quality!r}")
        code_path = Path(code.code_path)
        output_path = project_directory / mode / f"{mode}.mp4"
        media_directory = project_directory / mode / "manim_media_subprocess"
        usage_path = project_directory / "logs" / f"voice_usage_{uuid.uuid4().hex}.jsonl"
        raw_log_path = project_directory / "logs" / f"{mode}_{code_path.stem}.log"
        previous_subtitles = (
            {
                path.resolve(): (path.stat().st_mtime_ns, path.stat().st_size)
                for path in media_directory.rglob("*.srt")
            }
            if media_directory.is_dir()
            else {}
        )
        environment = build_render_environment(os.environ)
        source_root = str(Path(__file__).resolve().parents[2])
        python_path = environment.get("PYTHONPATH", "")
        environment["PYTHONPATH"] = (
            source_root if not python_path else f"{source_root}{os.pathsep}{python_path}"
        )
        if voiceover_enabled:
            _apply_voice_environment(
                environment,
                voice_provider=voice_provider,
                voice_model=voice_model,
                voice=voice,
                voice_language=voice_language,
                voice_speed=voice_speed,
                voice_prompt_template=voice_prompt_template,
                video_mode=mode,
                api_key=api_key,
            )
            environment[USAGE_PATH_ENV] = str(usage_path)

        command = [
            sys.executable,
            "-m",
            "manim",
            str(code_path),
            code.scene_name,
            _QUALITY_FLAGS[quality],
            "--media_dir",
            str(media_directory),
            "--output_file",
            mode,
        ]
        try:
            completed = self.command_runner(
                command,
                cwd=project_directory,
                env=environment,
                cancellation_check=self.cancellation_check,
                timeout_seconds=self.timeout_seconds,
            )
        except OSError as exc:
            _atomic_write_text(
                raw_log_path,
                redact(f"Não foi possível iniciar o Manim: {exc}", [api_key]),
            )
            return RenderResult(
                mode=mode,
                success=False,
                return_code=1,
                code_path=str(code_path),
                stderr="Não foi possível iniciar o processo isolado do Manim.",
                error_traceback=f"Subprocesso: {exc}",
                raw_log_path=str(raw_log_path),
                attempts=1,
                quality=quality,
            )
        except Exception:
            usage_path.unlink(missing_ok=True)
            raise

        usage_events = read_usage_events(usage_path)
        usage_path.unlink(missing_ok=True)
        raw_output = redact(
            f"STDOUT\n{completed.stdout}\n\nSTDERR\n{completed.stderr}",
            [api_key],
        )
        _atomic_write_text(raw_log_path, raw_output)
        if completed.returncode != 0:
            render_stderr = completed.stderr or completed.stdout
            return RenderResult(
                mode=mode,
                success=False,
                return_code=completed.returncode,
                code_path=str(code_path),
                stdout=completed.stdout,
                stderr=redact(classify_render_error(render_stderr), [api_key]),
                attempts=1,
                quality=quality,
                usage_events=usage_events,
                raw_log_path=str(raw_log_path),
            )

        candidates = sorted(
            media_directory.rglob(f"{mode}*.mp4"),
            key=lambda path: path.stat().st_mtime,
        )
        if not candidates:
            return RenderResult(
                mode=mode,
                success=False,
                return_code=1,
                code_path=str(code_path),
                stdout=completed.stdout,
                stderr="O processo isolado do Manim terminou sem produzir um vídeo.",
                attempts=1,
                quality=quality,
                usage_events=usage_events,
                raw_log_path=str(raw_log_path),
            )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(candidates[-1], output_path)
        subtitle_path = ""
        subtitle_candidates = sorted(
            (
                path
                for path in media_directory.rglob("*.srt")
                if previous_subtitles.get(path.resolve())
                != (path.stat().st_mtime_ns, path.stat().st_size)
            ),
            key=lambda path: path.stat().st_mtime_ns,
        )
        if subtitle_candidates:
            subtitle_output = project_directory / mode / f"{mode}.srt"
            shutil.copy2(subtitle_candidates[-1], subtitle_output)
            subtitle_path = str(subtitle_output)
        return RenderResult(
            mode=mode,
            success=True,
            return_code=0,
            video_path=str(output_path),
            code_path=str(code_path),
            stdout="Renderização Manim concluída em processo isolado.",
            attempts=1,
            quality=quality,
            usage_events=usage_events,
            subtitle_path=subtitle_path,
            raw_log_path=str(raw_log_path),
        )


def build_render_environment(source: Mapping[str, str]) -> dict[str, str]:
    """Copy only runtime variables required by Manim, never provider secrets."""
    return {key: value for key, value in source.items() if key in _RENDER_ENV_ALLOWLIST and value}


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary.write(content)
            temporary.flush()
            os.fsync(temporary.fileno())
            temporary_path = Path(temporary.name)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _run_cancellable_process(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    cancellation_check: Callable[[], None],
    timeout_seconds: int,
) -> subprocess.CompletedProcess[str]:
    """Run Manim while enforcing cancellation and a wall-clock timeout."""
    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    started_at = time.monotonic()
    while True:
        try:
            stdout, stderr = process.communicate(timeout=0.5)
        except subprocess.TimeoutExpired:
            try:
                cancellation_check()
            except Exception:
                _stop_process(process)
                raise
            if time.monotonic() - started_at >= timeout_seconds:
                _stop_process(process)
                return subprocess.CompletedProcess(
                    command,
                    returncode=124,
                    stdout="",
                    stderr=(
                        "A renderização excedeu o limite de "
                        f"{timeout_seconds} segundos e foi encerrada."
                    ),
                )
            continue
        return subprocess.CompletedProcess(
            command,
            returncode=process.returncode,
            stdout=stdout,
            stderr=stderr,
        )


def _stop_process(process: subprocess.Popen[str]) -> None:
    """Terminate a Manim process and escalate to kill when necessary."""
    process.terminate()
    try:
        process.communicate(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.communicate()


def _node_name(node: ast.expr) -> str:
    """Return the terminal identifier represented by an AST expression."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def classify_render_error(error_output: str) -> str:
    """Return a concise technical diagnostic; the complete output is stored separately."""
    for line in reversed(error_output.splitlines()):
        if "VoiceoverProviderError:" in line:
            return line.split("VoiceoverProviderError:", 1)[1].strip()
    normalized = error_output.casefold()
    if "syntaxerror" in normalized or "indentationerror" in normalized:
        category = "Erro de sintaxe"
    elif "modulenotfounderror" in normalized or "importerror" in normalized:
        category = "Erro de importação"
    elif "filenotfounderror" in normalized or "no such file or directory" in normalized:
        category = "Erro de caminho de arquivo"
    elif any(
        marker in normalized
        for marker in ("attributeerror", "unexpected keyword argument", "has no attribute")
    ):
        category = "Erro de API do Manim"
    else:
        category = "Erro de renderização do Manim"
    lines = [line.rstrip() for line in error_output.strip().splitlines() if line.strip()]
    relevant = "\n".join(lines[-80:])
    if len(relevant) > 6000:
        relevant = "…\n" + relevant[-5998:]
    return f"{category}:\n{relevant}"


def _apply_voice_environment(
    environment: dict[str, str],
    *,
    voice_provider: str,
    voice_model: str,
    voice: str,
    voice_language: str,
    voice_speed: float,
    voice_prompt_template: str,
    video_mode: str,
    api_key: str,
) -> None:
    """Set only the credentials and voice settings selected for one render."""
    environment["OLYMPIANIM_VOICE_PROVIDER"] = voice_provider
    environment["OLYMPIANIM_VOICE_MODEL"] = voice_model
    environment["OLYMPIANIM_VOICE"] = voice
    environment["OLYMPIANIM_VOICE_LANGUAGE"] = voice_language
    environment["OLYMPIANIM_VOICE_SPEED"] = str(voice_speed)
    environment["OLYMPIANIM_VOICE_PROMPT_TEMPLATE"] = voice_prompt_template
    environment["OLYMPIANIM_VIDEO_MODE"] = video_mode
    if voice_provider == "Google":
        environment["GOOGLE_API_KEY"] = api_key
    else:
        environment["OPENAI_API_KEY"] = api_key
