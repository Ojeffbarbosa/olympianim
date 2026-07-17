"""Native LangGraph workflow with explicit human review interrupts."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from olympianim.config import MAX_RENDER_RETRIES
from olympianim.database.repository import ProjectRepository
from olympianim.manim.presentation import (
    PresentationRenderer,
    check_generated_code_safety,
    prepare_source_watermark_code,
    prepare_voiceover_code,
    presentation_scene_name,
    strip_source_watermark_code,
)
from olympianim.prompts.defaults import WORKFLOW_PROMPT_NAMES
from olympianim.prompts.service import PromptService
from olympianim.schemas.llm import ManimCodeOutput
from olympianim.schemas.render import ManimCodeResult, VoiceConfig
from olympianim.services.artifact_service import ArtifactService
from olympianim.services.color_palette import ColorPaletteService
from olympianim.services.image_asset_service import AnimationAsset, ImageAssetService
from olympianim.services.llm_service import LLMImage, LLMRequest, LLMService
from olympianim.services.project_logging import ProjectLogger, ProjectToolCallback
from olympianim.services.subtitle_service import SubtitleService
from olympianim.services.usage_service import UsageContext, UsageService
from olympianim.services.video_assembler import VideoAssembler
from olympianim.tools import search_manim_reference

VideoMode = Literal["presentation", "solution"]
_ENVIRONMENT_ERROR_MARKERS = (
    "manim não está instalado",
    "no module named 'manim'",
    "ffmpeg not found",
    "libcairo",
)
_CODE_RESPONSE_FORMAT_REVISION = "manim-code-provider-strategy-v3"
MANUAL_PRESENTATION_RECOVERY_ENTRY = "recover_manual_presentation"


class _GeneratedCodeValidationError(RuntimeError):
    """Raised when a model response cannot safely enter workflow state or cache."""


def extract_python_code(content: str) -> str:
    """Extract Python source without altering the source itself.

    Code-producing agents are instructed to return plain Python, but some
    providers still wrap a complete file in a Markdown fence and append an
    explanation.  In that case, select the fenced source and leave syntax
    validation to ``check_generated_code_safety``.
    """
    lines = content.lstrip("\ufeff").strip().splitlines()
    fenced_blocks: list[tuple[str, list[str]]] = []
    index = 0

    while index < len(lines):
        stripped = lines[index].strip()
        if not stripped.startswith("```"):
            index += 1
            continue

        language = stripped[3:].strip().lower()
        closing_index = next(
            (
                candidate
                for candidate in range(index + 1, len(lines))
                if lines[candidate].strip() == "```"
            ),
            None,
        )
        if closing_index is None:
            break
        if language in {"", "python", "py"}:
            fenced_blocks.append((language, lines[index + 1 : closing_index]))
        index = closing_index + 1

    for accepted_languages in ({"python", "py"}, {""}):
        for language, block_lines in fenced_blocks:
            if language in accepted_languages:
                return "\n".join(block_lines).strip()

    # Preserve compatibility with an opening fence that the provider forgot
    # to close.  Any trailing prose remains visible to the syntax safeguard.
    if lines:
        first_line = lines[0].strip()
        first_language = first_line[3:].strip().lower() if first_line.startswith("```") else None
        if first_language in {"", "python", "py"}:
            lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]

    return "\n".join(lines).strip()


def voiceover_prompt_requirements(
    voice: VoiceConfig,
    *,
    synchronize_visuals: bool = True,
) -> str:
    """Return the app-owned voice contract only when narration is enabled."""
    if not voice.enabled:
        return ""
    base = """Integração de voz controlada pelo Olympianim:
- use exclusivamente ``from olympianim.manim.voiceover import ConfiguredVoiceoverScene`` e herde dessa classe;
- use blocos ``with self.voiceover(text=...)`` para sincronizar a narração;
- o gerenciador ``with self.voiceover(...)`` controla todo o ciclo da fala e, ao sair do bloco, espera automaticamente o tempo de áudio restante; depois da última animação, basta encerrar o bloco para manter o quadro estável até o fim da fala;
- nunca chame nem redefina ``wait_for_voiceover``: esse método já é herdado pela cena e deve conservar a assinatura da biblioteca;
- além de ``construct``, não sobrescreva métodos herdados de ``ConfiguredVoiceoverScene``, ``VoiceoverScene`` ou ``Scene``; helpers próprios são permitidos somente com nomes descritivos que não existam nas classes-base;
- use ``tracker.duration`` apenas para dimensionar ``run_time`` ou pausas explicitamente necessárias dentro da fala, sem substituir o controle automático do bloco ``with``;
- use esses blocos somente com fala não vazia; para movimento sem fala, execute ``self.play(...)`` diretamente;
- não importe ``manim_voiceover`` nem qualquer outro símbolo de ``olympianim``;
- não configure serviços, provedores, credenciais ou vozes e não pesquise essa camada com ``search_manim_reference``.
"""
    if not synchronize_visuals:
        return base
    return base + """Sincronização didática obrigatória:
- implemente cada unidade ``contexto visual → fala → evento visual`` do plano como um bloco curto e atômico;
- antes de cada fala, distinga **contexto de acompanhamento** de **informação construída**: coloque com ``self.add(...)`` ou movimento sem fala o enunciado, diagrama-base, configuração inicial, objetos comparados e dados fornecidos que o aluno precisa observar; não prolongue o aparecimento desse contexto com ``tracker.duration``;
- durante a fala, revele somente a informação construída naquele momento, como destaque, marcação interpretativa, transformação, relação inferida, operação, resultado ou resposta;
- quando a narração percorre partes de um mesmo objeto, mantenha o objeto completo visível e sincronize apenas os destaques; construa-o por partes somente quando a construção for o próprio raciocínio;
- não reúna em um único bloco uma explicação longa com vários ``self.play`` rápidos; divida a fala no ponto em que muda o evento visual;
- faça remoções, reposicionamentos e limpeza da tela antes da fala correspondente ou em movimento sem fala, para que não consumam o início da narração de uma nova ideia;
- para uma única animação que deve acompanhar toda a fala curta, dimensione seu ``run_time`` com ``tracker.duration``; não reutilize a duração total em várias animações sequenciais;
- quando um bloco curto precisar de marcos internos, insira ``<bookmark mark='nome_unico'/>`` imediatamente antes das palavras que apresentam o elemento: use ``self.wait_until_bookmark('nome_unico')`` para iniciar o evento nesse ponto;
- use ``tracker.time_until_bookmark('proximo_marco')`` somente antes desse marco, como duração de uma animação que deve terminar nele; não use ``time_until_bookmark`` e ``wait_until_bookmark`` para o mesmo evento;
- mantenha nomes de bookmarks únicos no bloco, formados por letras, números ou sublinhado, e não crie bookmarks sem uma animação sincronizada correspondente;
- em cálculos, revele na ordem narrada os dados ou operandos, a operação ou transformação e, por último, o resultado; nunca mostre uma igualdade final, resposta ou consequência antes da frase que a enuncia;
- antes de concluir, confirme no código que o contexto necessário já está visível, nenhuma informação construída aparece antes de ser explicada e nenhuma fala longa permanece com todas as animações concentradas nos primeiros segundos.
"""


def _model_selection_update(decision: Mapping[str, Any]) -> dict[str, str]:
    """Extract the model selected at a human-review boundary."""
    raw_selection = decision.get("model_selection")
    if raw_selection is None:
        return {}
    if not isinstance(raw_selection, Mapping):
        raise ValueError("Seleção de modelo inválida ao retomar o fluxo.")
    provider = str(raw_selection.get("provider", "")).strip()
    model = str(raw_selection.get("model", "")).strip()
    if not provider or not model:
        raise ValueError("Provedor e modelo são obrigatórios ao retomar o fluxo.")
    return {"llm_provider": provider, "llm_model": model}


def _operation_update(decision: Mapping[str, Any]) -> dict[str, str]:
    """Persist which explicit operation consumed the current interrupt."""
    operation_id = str(decision.get("operation_id", "")).strip()
    return {"last_applied_operation_id": operation_id} if operation_id else {}


class ApprovalState(TypedDict, total=False):
    workflow_entry: str
    project_id: str
    problem_statement: str
    teacher_solution: str
    teacher_instructions: str
    llm_provider: str
    llm_model: str
    voice: dict[str, Any]
    voice_prompt_template: str
    color_palette_snapshot: str
    mode: VideoMode
    plan: str
    code: str
    render_code: str
    code_path: str
    solution_basis: str
    solution_basis_source: str
    solution_basis_approved: bool
    render_path: str
    presentation_render_path: str
    solution_render_path: str
    final_render_path: str
    render_error: str
    retry_count: int
    phase: str
    last_applied_operation_id: str
    presentation_subtitle_path: str
    solution_subtitle_path: str
    final_subtitle_path: str


def build_approved_workflow(
    *,
    llm_api_key: str,
    voice_api_key: str,
    llm_service: LLMService,
    prompt_service: PromptService,
    artifact_service: ArtifactService,
    renderer: PresentationRenderer,
    repository: ProjectRepository,
    project_id: str,
    checkpointer: Any,
    cancellation_check: Callable[[], None] | None = None,
    progress_callback: Callable[[str, int], None] | None = None,
    execution_id: str = "",
    operation_id: str = "",
    workflow_revision: int = 1,
    video_assembler: VideoAssembler | None = None,
) -> Any:
    """Build one resumable graph; credentials live only in this invocation closure."""
    project_logger = ProjectLogger(
        repository,
        project_id,
        secrets=(llm_api_key, voice_api_key),
    )
    image_asset_service = ImageAssetService(
        repository=repository,
        projects_dir=artifact_service.projects_dir,
    )
    animation_assets = image_asset_service.list_assets(project_id)
    project_files = repository.list_generated_files(project_id)
    problem_images = tuple(
        LLMImage.from_path(item.path, label=f"Imagem do enunciado {item.version}")
        for item in project_files
        if item.file_type == "problem_image"
    )
    solution_images = tuple(
        LLMImage.from_path(item.path, label=f"Imagem da solução {item.version}")
        for item in project_files
        if item.file_type == "solution_image"
    )
    check_cancelled = cancellation_check or (lambda: None)
    report_progress = progress_callback or (lambda _step, _progress: None)
    usage_service = UsageService(repository)
    assembler = video_assembler or VideoAssembler()

    def prepare_generated_code(
        content: str,
        *,
        voice: VoiceConfig,
        strip_watermark: bool = False,
    ) -> str:
        """Normalize and validate model code before it becomes cacheable state."""
        code = extract_python_code(content)
        code = prepare_voiceover_code(code, require_voiceover=voice.enabled)
        if strip_watermark:
            code = strip_source_watermark_code(code)
        errors = check_generated_code_safety(
            code,
            require_voiceover=voice.enabled,
            allowed_image_paths=_allowed_image_paths(animation_assets),
        )
        if errors:
            raise _GeneratedCodeValidationError(" ".join(errors))
        return code

    def call(
        role: str,
        mode: VideoMode,
        values: dict[str, object],
        *,
        response_processor: Callable[[str], str] | None = None,
    ) -> str:
        name = WORKFLOW_PROMPT_NAMES[(role, mode)]
        agent = {
            "planner": "workflow_planner",
            "builder": "workflow_builder",
            "solver": "solution_solver",
            "debugger": "workflow_debugger",
        }[role]
        prompt = next(
            item for item in prompt_service.list_prompts(agent) if item.prompt.name == name
        )
        context = build_animation_asset_context(animation_assets, role)
        template_text = prompt.latest_version.template_text
        if (
            role == "planner"
            and mode == "presentation"
            and "{solution_basis}" not in template_text
        ):
            template_text += """

Use a base abaixo somente como referência privada para alinhar perguntas e representações. Não revele resposta, caminho decisivo nem mencione esta referência no vídeo.

<referencia_privada_da_solucao>
{solution_basis}
</referencia_privada_da_solucao>
"""
        palette_context = ColorPaletteService.prompt_context(
            str(values.pop("color_palette_snapshot", ""))
        )
        include_solution_images = role == "solver" or role == "planner" or mode == "solution"
        images = problem_images + (solution_images if include_solution_images else ())
        if problem_images:
            template_text += (
                "\n\nAs imagens do enunciado foram anexadas na ordem de leitura. "
                "Leia e interprete diretamente seu conteúdo visual; quando o texto do "
                "enunciado estiver vazio, transcreva e use o enunciado presente nelas."
            )
        if role == "solver" and solution_images:
            template_text += (
                "\n\nAs imagens da solução do professor vêm depois das imagens do "
                "enunciado e estão na ordem de leitura. Elas são a fonte vinculante do "
                "método: siga a mesma estratégia e a mesma ordem lógica, preservando "
                "explicitamente qualquer teorema, regra, desigualdade, construção ou "
                "algoritmo usado. Você pode completar justificativas omitidas, mas não "
                "pode trocar o método por uma solução alternativa, ainda que equivalente. "
                "Se um passo essencial estiver ilegível ou inconsistente, indique esse "
                "ponto para revisão humana em vez de substituí-lo silenciosamente. Antes "
                "de responder, confirme que todas as técnicas usadas nas imagens também "
                "aparecem no texto produzido."
            )
        elif role == "planner" and mode == "presentation" and solution_images:
            template_text += (
                "\n\nAs imagens da solução do professor vêm depois das imagens do "
                "enunciado e estão na ordem de leitura. Consulte-as somente como "
                "referência privada para alinhar perguntas e representações; não revele "
                "seu conteúdo, resultado ou caminho decisivo no vídeo de apresentação."
            )
        elif mode == "solution" and solution_images:
            template_text += (
                "\n\nAs imagens da solução do professor vêm depois das imagens do "
                "enunciado e estão na ordem de leitura. Use essa solução como referência."
            )
        if context:
            template_text = f"{template_text}\n\n{{animation_assets_context}}"
            values["animation_assets_context"] = context
        if role in {"builder", "debugger"} and palette_context:
            template_text = f"{template_text}\n\n{{color_palette_context}}"
            values["color_palette_context"] = palette_context
        provider = str(values.pop("provider"))
        model = str(values.pop("model"))
        request = LLMRequest(
            provider=provider,
            model=model,
            api_key=llm_api_key,
            template_text=template_text,
            prompt_values=values,
            temperature=0.0 if role != "planner" else 0.2,
            images=images,
        )
        prompt_render = LLMService(prompt_service=prompt_service).render_prompt(request)
        stable_operation_id = operation_id or execution_id or "interactive"
        cache_identity = "\0".join(
            (
                stable_operation_id,
                role,
                mode,
                str(workflow_revision),
                provider,
                model,
                prompt_render.prompt_sha256,
                (
                    _CODE_RESPONSE_FORMAT_REVISION
                    if role in {"builder", "debugger"}
                    else "plain-text-v1"
                ),
            )
        )
        cache_key = hashlib.sha256(cache_identity.encode("utf-8")).hexdigest()
        repository.record_project_prompt(
            project_id,
            agent_type=prompt.prompt.agent_type,
            prompt_id=prompt.prompt.id,
            prompt_version=prompt.latest_version.version,
            rendered_prompt_snapshot=prompt_render.rendered_prompt,
            rendered_system_snapshot=prompt_render.system_prompt,
            rendered_user_snapshot=prompt_render.user_prompt,
            prompt_sha256=prompt_render.prompt_sha256,
            operation_id=stable_operation_id,
        )
        cached = repository.get_llm_call_cache(cache_key)
        if cached is not None:
            project_logger.info(
                f"agent.{role}.{mode}",
                "Resposta concluída anteriormente reutilizada após retomada.",
            )
            repository.add_workflow_event(
                project_id,
                event_key=f"{cache_key}:llm.reused",
                event_type="llm.reused",
                operation_id=stable_operation_id,
                job_id=execution_id,
                phase=f"{role}_{mode}",
                payload=json.dumps(
                    {
                        "provider": cached.provider,
                        "requested_model": cached.requested_model,
                        "resolved_model": cached.resolved_model,
                        "prompt_sha256": cached.prompt_sha256,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            )
            return (
                response_processor(cached.response_text)
                if response_processor is not None
                else cached.response_text
            )
        callbacks = (
            (ProjectToolCallback(project_logger, role=role, mode=mode),)
            if role in {"builder", "debugger"}
            else ()
        )
        request = LLMRequest(
            provider=provider,
            model=model,
            api_key=llm_api_key,
            template_text=template_text,
            prompt_values=values,
            temperature=0.0 if role != "planner" else 0.2,
            callbacks=callbacks,
            usage_context=UsageContext(
                project_id=project_id,
                execution_id=execution_id,
                call_key=cache_key,
                agent_type=role,
                stage=mode,
            ),
            images=images,
        )
        step = f"agent.{role}.{mode}"
        project_logger.info(step, f"Chamada iniciada com {provider}:{model}.")
        result = (
            llm_service.call_agent(
                request,
                tools=(search_manim_reference,),
                response_schema=ManimCodeOutput,
            )
            if role in {"builder", "debugger"}
            else llm_service.call_text(request)
        )
        if not result.result.ok:
            project_logger.error(step, result.result.message)
            raise RuntimeError(result.result.message)
        content = result.result.content.strip()
        processed_content = (
            response_processor(content) if response_processor is not None else content
        )
        repository.save_llm_call_cache(
            cache_key=cache_key,
            project_id=project_id,
            operation_id=stable_operation_id,
            role=role,
            mode=mode,
            provider=result.result.provider or provider,
            requested_model=model,
            resolved_model=getattr(result.result, "resolved_model", "") or result.result.model,
            finish_reason=getattr(result.result, "finish_reason", ""),
            prompt_sha256=prompt_render.prompt_sha256,
            response_text=content,
        )
        repository.add_workflow_event(
            project_id,
            event_key=f"{cache_key}:llm.completed",
            event_type="llm.completed",
            operation_id=stable_operation_id,
            job_id=execution_id,
            phase=f"{role}_{mode}",
            payload=json.dumps(
                {
                    "provider": result.result.provider or provider,
                    "requested_model": model,
                    "resolved_model": (
                        getattr(result.result, "resolved_model", "") or result.result.model
                    ),
                    "finish_reason": getattr(result.result, "finish_reason", ""),
                    "prompt_sha256": prompt_render.prompt_sha256,
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
        )
        project_logger.info(step, "Chamada concluída com sucesso.")
        return processed_content

    def prepare_solution_basis(state: ApprovalState) -> Command[str]:
        """Establish one mathematical source before presentation planning."""
        check_cancelled()
        report_progress("prepare_solution_basis", 20)
        teacher_basis = state.get("teacher_solution", "").strip()
        if teacher_basis:
            project_logger.info(
                "solution_basis.teacher",
                "Solução textual do professor adotada como base matemática aprovada.",
            )
            return Command(
                update={
                    "solution_basis": teacher_basis,
                    "solution_basis_source": "teacher_text",
                    "solution_basis_approved": True,
                    "phase": "plan_presentation",
                },
                goto="plan_presentation",
            )

        source = "teacher_image_interpretation" if solution_images else "ai_solution"
        basis = call(
            "solver",
            "solution",
            {
                "provider": state["llm_provider"],
                "model": state["llm_model"],
                "problem_statement": state["problem_statement"],
                "teacher_instructions": state.get("teacher_instructions", ""),
            },
        )
        if not basis.strip():
            raise RuntimeError("O solucionador não produziu uma base matemática para revisão.")
        project_logger.info(
            "solution_basis.generated",
            (
                "Interpretação da solução anexada preparada para revisão."
                if solution_images
                else "Solução matemática preparada pelo agente para revisão."
            ),
        )
        return Command(
            update={
                "solution_basis": basis,
                "solution_basis_source": source,
                "solution_basis_approved": False,
                "phase": "review_solution_basis",
            },
            goto="review_solution_basis",
        )

    def review_solution_basis(state: ApprovalState) -> Command[str]:
        """Require approval for every model-produced mathematical basis."""
        check_cancelled()
        report_progress("review_solution_basis", 90)
        decision = interrupt(
            {
                "kind": "solution_basis",
                "content": state["solution_basis"],
                "source": state.get("solution_basis_source", "ai_solution"),
                "phase": state["phase"],
            }
        )
        action = decision.get("action", "approve")
        model_update = _model_selection_update(decision)
        operation_update = _operation_update(decision)
        project_logger.info("review.solution_basis", f"Ação do usuário: {action}.")
        if action == "edit":
            edited_basis = str(decision["content"]).strip()
            if not edited_basis:
                raise ValueError("A base matemática não pode ficar vazia.")
            return Command(
                update={
                    **model_update,
                    **operation_update,
                    "solution_basis": edited_basis,
                    "solution_basis_approved": False,
                    "phase": "review_solution_basis",
                },
                goto="review_solution_basis",
            )
        if action == "regenerate":
            return Command(
                update={
                    **model_update,
                    **operation_update,
                    "solution_basis": "",
                    "solution_basis_approved": False,
                    "phase": "prepare_solution_basis",
                },
                goto="prepare_solution_basis",
            )
        if action != "approve":
            return Command(
                update={**model_update, **operation_update, "phase": "stopped"},
                goto=END,
            )
        if not state.get("solution_basis", "").strip():
            raise ValueError("A base matemática não pode ser aprovada vazia.")
        return Command(
            update={
                **model_update,
                **operation_update,
                "solution_basis_approved": True,
                "phase": "plan_presentation",
            },
            goto="plan_presentation",
        )

    def plan(state: ApprovalState, mode: VideoMode) -> dict[str, object]:
        check_cancelled()
        report_progress(f"plan_{mode}", 35)
        basis = state.get("solution_basis", "")
        if mode == "solution" and not basis:
            basis = state["teacher_solution"].strip() or call(
                "solver",
                "solution",
                {
                    "provider": state["llm_provider"],
                    "model": state["llm_model"],
                    "problem_statement": state["problem_statement"],
                    "teacher_instructions": state.get("teacher_instructions", ""),
                },
            )
        return {
            "mode": mode,
            "plan": call(
                "planner",
                mode,
                {
                    "provider": state["llm_provider"],
                    "model": state["llm_model"],
                    "problem_statement": state["problem_statement"],
                    "teacher_instructions": state.get("teacher_instructions", ""),
                    "solution_basis": basis,
                },
            ),
            "solution_basis": basis,
            "phase": f"review_plan_{mode}",
        }

    def review_plan(state: ApprovalState) -> Command[str]:
        check_cancelled()
        report_progress(f"review_plan_{state.get('mode', 'presentation')}", 90)
        decision = interrupt(
            {
                "kind": "plan",
                "mode": state["mode"],
                "content": state["plan"],
                "phase": state["phase"],
            }
        )
        action = decision.get("action", "approve")
        model_update = _model_selection_update(decision)
        operation_update = _operation_update(decision)
        project_logger.info(f"review.plan.{state['mode']}", f"Ação do usuário: {action}.")
        if action == "edit":
            return Command(
                update={
                    **model_update,
                    **operation_update,
                    "plan": str(decision["content"]),
                    "phase": state["phase"],
                },
                goto="review_plan",
            )
        if action == "regenerate":
            return Command(
                update={**model_update, **operation_update},
                goto="plan_presentation" if state["mode"] == "presentation" else "plan_solution",
            )
        if action != "approve":
            return Command(
                update={**model_update, **operation_update, "phase": "stopped"},
                goto=END,
            )
        return Command(
            update={
                **model_update,
                **operation_update,
                "phase": f"build_{state['mode']}",
            },
            goto="build",
        )

    def build(state: ApprovalState) -> dict[str, object]:
        check_cancelled()
        report_progress(f"build_{state.get('mode', 'presentation')}", 40)
        mode = state["mode"]
        voice = VoiceConfig(**state["voice"])
        code = call(
            "builder",
            mode,
            {
                "provider": state["llm_provider"],
                "model": state["llm_model"],
                "problem_statement": state["problem_statement"],
                "approved_plan": state["plan"],
                "voiceover_requirements": voiceover_prompt_requirements(voice),
                "color_palette_snapshot": state.get("color_palette_snapshot", ""),
            },
            response_processor=lambda content: prepare_generated_code(
                content,
                voice=voice,
            ),
        )
        return {"code": code, "phase": f"review_code_{mode}"}

    def review_code(state: ApprovalState) -> Command[str]:
        check_cancelled()
        report_progress(f"review_code_{state.get('mode', 'presentation')}", 90)
        decision = interrupt(
            {
                "kind": "code",
                "mode": state["mode"],
                "content": state["code"],
                "phase": state["phase"],
            }
        )
        action = decision.get("action", "approve")
        model_update = _model_selection_update(decision)
        operation_update = _operation_update(decision)
        project_logger.info(f"review.code.{state['mode']}", f"Ação do usuário: {action}.")
        if action == "edit":
            return Command(
                update={
                    **model_update,
                    **operation_update,
                    "code": str(decision["content"]),
                    "phase": state["phase"],
                },
                goto="review_code",
            )
        if action == "regenerate":
            return Command(update={**model_update, **operation_update}, goto="build")
        if action != "approve":
            return Command(
                update={**model_update, **operation_update, "phase": "stopped"},
                goto=END,
            )
        mode = state["mode"]
        project = repository.get_project(state["project_id"])
        canonical_code = strip_source_watermark_code(state["code"])
        render_code = prepare_source_watermark_code(
            canonical_code, project.problem_source if project is not None else ""
        )
        path = artifact_service.save_manim_code(
            state["project_id"], mode=mode, code=render_code, version=1
        )
        return Command(
            update={
                **model_update,
                **operation_update,
                "code": canonical_code,
                "render_code": render_code,
                "code_path": str(path),
                "phase": f"render_{mode}",
            },
            goto="render",
        )

    def render(state: ApprovalState) -> Command[str]:
        check_cancelled()
        report_progress(f"render_{state.get('mode', 'presentation')}", 45)
        mode = state["mode"]
        project_logger.info(f"render.{mode}", "Renderização iniciada.")
        voice = VoiceConfig(**state["voice"])
        render_code = state.get("render_code", state["code"])
        result = renderer.render(
            ManimCodeResult(
                mode=mode,
                scene_name=presentation_scene_name(render_code, require_voiceover=voice.enabled),
                code=render_code,
                code_path=state["code_path"],
            ),
            project_directory=artifact_service.project_directory(state["project_id"]),
            mode=mode,
            api_key=voice_api_key,
            voice_provider=voice.provider,
            voice_model=voice.model,
            voice=voice.voice,
            voice_language=voice.language,
            voice_speed=voice.speed,
            voice_prompt_template=state.get("voice_prompt_template", "{transcript}"),
            voiceover_enabled=voice.enabled,
            quality=repository.get_setting("render_quality", "low_quality"),
        )
        render_number = int(state.get("retry_count", 0)) + 1
        usage_service.record_speech_events(
            result.usage_events,
            project_id=state["project_id"],
            execution_id=execution_id,
            stage=mode,
            render_key=(f"{execution_id or 'interactive'}:voice:{mode}:render-{render_number}"),
        )
        artifact_version = _artifact_version(state["code_path"])
        if result.raw_log_path and Path(result.raw_log_path).is_file():
            artifact_service.register_existing(
                state["project_id"],
                file_type=f"{mode}_render_log",
                path=Path(result.raw_log_path),
                description=f"Saída técnica completa do render de {mode}.",
                version=artifact_version,
                artifact_key=f"{mode}_render_log:v{artifact_version}",
            )
        if result.success:
            project_logger.info(f"render.{mode}", "Renderização concluída com sucesso.")
            artifact_service.register_video(
                state["project_id"],
                mode=mode,
                video_path=Path(result.video_path),
                version=artifact_version,
            )
            subtitle_updates: dict[str, object] = {}
            if result.subtitle_path and Path(result.subtitle_path).is_file():
                subtitle_path = Path(result.subtitle_path)
                transcript_path = artifact_service.save_text(
                    state["project_id"],
                    relative_path=f"{mode}/{mode}.txt",
                    content=SubtitleService.transcript_file(subtitle_path),
                    file_type=f"{mode}_transcript",
                    description=f"Transcrição textual do vídeo de {mode}.",
                    version=artifact_version,
                    artifact_key=f"{mode}_transcript:v{artifact_version}",
                )
                artifact_service.register_subtitle(
                    state["project_id"],
                    mode=mode,
                    subtitle_path=subtitle_path,
                    transcript_path=transcript_path,
                    version=artifact_version,
                )
                subtitle_updates[f"{mode}_subtitle_path"] = str(subtitle_path)
            if mode == "presentation":
                return Command(
                    update={
                        **subtitle_updates,
                        "render_path": result.video_path,
                        "presentation_render_path": result.video_path,
                        "render_error": "",
                        "phase": "presentation_complete",
                    },
                    goto="review_presentation_complete",
                )
            updates: dict[str, object] = {
                **subtitle_updates,
                "render_path": result.video_path,
                "solution_render_path": result.video_path,
                "render_error": "",
                "phase": "completed",
            }
            project = repository.get_project(state["project_id"])
            if project is not None and project.output_delivery_mode == "combined":
                report_progress("assemble_final", 80)
                assembled = assembler.combine(
                    (
                        Path(state["presentation_render_path"]),
                        Path(result.video_path),
                    ),
                    artifact_service.project_directory(state["project_id"])
                    / "final"
                    / "olympianim_final.mp4",
                )
                if not assembled.success:
                    message = assembled.error_message or "Falha ao montar o vídeo único."
                    project_logger.error("assemble_final", message)
                    return Command(
                        update=updates | {"render_error": message, "phase": "failed"},
                        goto=END,
                    )
                final_path = Path(assembled.video_path)
                artifact_service.register_final_video(state["project_id"], final_path)
                project_logger.info("assemble_final", "Vídeo único montado com sucesso.")
                updates["final_render_path"] = str(final_path)
                presentation_srt = str(state.get("presentation_subtitle_path", ""))
                solution_srt = str(subtitle_updates.get("solution_subtitle_path", ""))
                probe_duration = getattr(assembler, "probe_duration", None)
                if (
                    presentation_srt
                    and solution_srt
                    and Path(presentation_srt).is_file()
                    and Path(solution_srt).is_file()
                    and callable(probe_duration)
                ):
                    duration = probe_duration(Path(state["presentation_render_path"]))
                    if duration is not None:
                        combined_srt = SubtitleService.combine(
                            Path(presentation_srt).read_text(encoding="utf-8-sig"),
                            Path(solution_srt).read_text(encoding="utf-8-sig"),
                            offset_seconds=float(duration),
                        )
                        final_subtitle = artifact_service.save_text(
                            state["project_id"],
                            relative_path="final/olympianim_final.srt",
                            content=combined_srt,
                            file_type="final_subtitle",
                            description="Legendas combinadas da apresentação e resolução.",
                            artifact_key="final_subtitle:v1",
                        )
                        artifact_service.save_text(
                            state["project_id"],
                            relative_path="final/olympianim_final.txt",
                            content=SubtitleService.transcript(combined_srt),
                            file_type="final_transcript",
                            description="Transcrição combinada dos dois vídeos.",
                            artifact_key="final_transcript:v1",
                        )
                        updates["final_subtitle_path"] = str(final_subtitle)
            return Command(update=updates, goto=END)
        error = result.stderr or result.error_traceback
        project_logger.error(f"render.{mode}", error or "Falha sem diagnóstico.")
        if state.get("retry_count", 0) >= MAX_RENDER_RETRIES or _is_non_code_render_error(error):
            return Command(
                update={"render_error": error, "phase": "failed"},
                goto=END,
            )
        return Command(
            update={"render_error": error, "phase": f"repair_{mode}"},
            goto="repair",
        )

    def repair(state: ApprovalState) -> Command[str]:
        check_cancelled()
        report_progress(f"repair_{state.get('mode', 'presentation')}", 55)
        """Repair code automatically; human approval already covered the initial draft."""
        mode = state["mode"]
        voice = VoiceConfig(**state["voice"])
        try:
            code = call(
                "debugger",
                mode,
                {
                    "provider": state["llm_provider"],
                    "model": state["llm_model"],
                    "manim_code": strip_source_watermark_code(state["code"]),
                    "render_error": state["render_error"],
                    "voiceover_requirements": voiceover_prompt_requirements(
                        voice,
                        synchronize_visuals=False,
                    ),
                    "color_palette_snapshot": state.get("color_palette_snapshot", ""),
                },
                response_processor=lambda content: prepare_generated_code(
                    content,
                    voice=voice,
                    strip_watermark=True,
                ),
            )
        except _GeneratedCodeValidationError as exc:
            return Command(update={"render_error": str(exc), "phase": "failed"}, goto=END)
        project = repository.get_project(state["project_id"])
        render_code = prepare_source_watermark_code(
            code, project.problem_source if project is not None else ""
        )
        retry_count = state.get("retry_count", 0) + 1
        path = artifact_service.save_manim_code(
            state["project_id"], mode=mode, code=render_code, version=retry_count + 1
        )
        return Command(
            update={
                "code": code,
                "render_code": render_code,
                "code_path": str(path),
                "retry_count": retry_count,
                "phase": f"render_{mode}",
            },
            goto="render",
        )

    def review_presentation_complete(state: ApprovalState) -> Command[str]:
        check_cancelled()
        report_progress("presentation_complete", 90)
        decision = interrupt(
            {
                "kind": "presentation_complete",
                "video_path": state.get("presentation_render_path", state["render_path"]),
                "phase": state["phase"],
            }
        )
        action = decision.get("action", "")
        model_update = _model_selection_update(decision)
        operation_update = _operation_update(decision)
        project_logger.info(
            "review.presentation_complete",
            f"Ação do usuário: {action or 'inválida'}.",
        )
        if action != "generate_solution":
            raise ValueError("A apresentação aguarda a ação 'generate_solution'.")
        return Command(
            update={
                **model_update,
                **operation_update,
                "workflow_entry": "",
                "phase": "plan_solution",
                "retry_count": 0,
            },
            goto="plan_solution",
        )

    def route_start(state: ApprovalState) -> str:
        """Enter the normal workflow or restore its presentation review boundary."""
        if state.get("workflow_entry") == MANUAL_PRESENTATION_RECOVERY_ENTRY:
            return "review_presentation_complete"
        return "prepare_solution_basis"

    graph = StateGraph(ApprovalState)
    graph.add_node("prepare_solution_basis", prepare_solution_basis)
    graph.add_node("review_solution_basis", review_solution_basis)
    graph.add_node("plan_presentation", lambda state: plan(state, "presentation"))
    graph.add_node("plan_solution", lambda state: plan(state, "solution"))
    graph.add_node("review_plan", review_plan)
    graph.add_node("build", build)
    graph.add_node("review_code", review_code)
    graph.add_node("render", render)
    graph.add_node("repair", repair)
    graph.add_node("review_presentation_complete", review_presentation_complete)
    graph.add_conditional_edges(
        START,
        route_start,
        {
            "prepare_solution_basis": "prepare_solution_basis",
            "review_presentation_complete": "review_presentation_complete",
        },
    )
    graph.add_edge("plan_presentation", "review_plan")
    graph.add_edge("plan_solution", "review_plan")
    graph.add_edge("build", "review_code")
    return graph.compile(checkpointer=checkpointer)


def _is_environment_error(error: str) -> bool:
    normalized = error.casefold()
    return any(marker in normalized for marker in _ENVIRONMENT_ERROR_MARKERS)


def _artifact_version(path: str) -> int:
    match = re.search(r"_v(\d+)\.py$", path)
    return int(match.group(1)) if match else 1


def _allowed_image_paths(assets: tuple[AnimationAsset, ...]) -> frozenset[str]:
    return frozenset(asset.manim_path for asset in assets)


def build_animation_asset_context(
    assets: tuple[AnimationAsset, ...],
    role: str,
) -> str:
    """Build the optional prompt suffix for image-aware workflow roles."""
    if not assets or role not in {"planner", "builder", "debugger"}:
        return ""
    lines = [
        "RECURSOS VISUAIS ANEXADOS PELO PROFESSOR:",
        "Use-os somente quando contribuírem para a explicação e não invente arquivos.",
    ]
    for asset in assets:
        lines.append(f"- arquivo: {asset.filename}")
        lines.append(f"  descrição: {asset.description}")
        lines.append("  fundo transparente: " + ("sim" if asset.background_removed else "não"))
        if role in {"builder", "debugger"}:
            lines.append(f"  caminho Manim autorizado: {asset.manim_path}")
    if role in {"builder", "debugger"}:
        lines.append(
            "Para exibir um recurso, use ImageMobject com exatamente o caminho autorizado."
        )
    return "\n".join(lines)


def _is_non_code_render_error(error: str) -> bool:
    """Return whether retrying or editing Manim code cannot fix the failure."""
    normalized = error.casefold()
    return _is_environment_error(error) or any(
        marker in normalized
        for marker in (
            "voiceoverprovidererror",
            "narração via google falhou",
            "narração via openai falhou",
            "gemini tts (",
            "chave da api",
        )
    )
