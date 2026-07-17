"""End-to-end tests for the approved LangGraph workflow."""

from __future__ import annotations

import ast
import sqlite3
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from olympianim.database.models import ProjectCreate
from olympianim.database.repository import ProjectRepository
from olympianim.graph.approved_workflow import (
    MANUAL_PRESENTATION_RECOVERY_ENTRY,
    build_approved_workflow,
    extract_python_code,
    voiceover_prompt_requirements,
)
from olympianim.prompts.service import PromptService
from olympianim.providers.llm.base import LLMCallResult
from olympianim.schemas.llm import ManimCodeOutput
from olympianim.schemas.render import RenderResult, VoiceConfig
from olympianim.services.artifact_service import ArtifactService
from olympianim.services.video_assembler import VideoAssemblyResult

_PRESENTATION_CODE = """from manim import *

class PresentationScene(Scene):
    def construct(self):
        self.add(Text("Leia o problema"))
"""

_SOLUTION_CODE = """from manim import *

class SolutionScene(Scene):
    def construct(self):
        self.add(Text("Resolucao"))
"""


def test_extract_python_code_preserves_plain_source() -> None:
    assert extract_python_code(_PRESENTATION_CODE) == _PRESENTATION_CODE.strip()


def test_extract_python_code_ignores_explanation_after_python_fence() -> None:
    response = f"""```python
{_PRESENTATION_CODE.rstrip()}
```

Implementei a sequência contexto visual → fala → evento visual.
"""

    assert extract_python_code(response) == _PRESENTATION_CODE.strip()


def test_extract_python_code_accepts_leading_prose_and_unlabelled_fence() -> None:
    response = f"""Segue o arquivo solicitado:

```
{_SOLUTION_CODE.rstrip()}
```

Fim da resposta.
"""

    assert extract_python_code(response) == _SOLUTION_CODE.strip()


def test_extract_python_code_prefers_explicit_python_fence() -> None:
    response = f"""```
trecho ilustrativo
```

```python
{_PRESENTATION_CODE.rstrip()}
```
"""

    assert extract_python_code(response) == _PRESENTATION_CODE.strip()


def test_extract_python_code_accepts_unclosed_initial_fence() -> None:
    response = f"""```python
{_SOLUTION_CODE.rstrip()}
"""

    assert extract_python_code(response) == _SOLUTION_CODE.strip()


def test_extract_python_code_does_not_rewrite_invalid_python_symbols() -> None:
    invalid_source = "resultado → valor"
    extracted = extract_python_code(invalid_source)

    assert extracted == invalid_source
    try:
        ast.parse(extracted)
    except SyntaxError as exc:
        assert "invalid character '→'" in str(exc)
    else:
        raise AssertionError("O símbolo inválido não pode ser corrigido silenciosamente.")


class FakeLLMService:
    """Return deterministic role-specific artifacts without external APIs."""

    def __init__(self) -> None:
        self.agent_requests: list[Any] = []
        self.agent_response_schemas: list[object] = []
        self.text_requests: list[Any] = []

    def call_text(self, request: Any) -> SimpleNamespace:
        self.text_requests.append(request)
        template = request.template_text
        if "Produza uma base matemática única" in template:
            content = "## Solução\nUma solução completa e justificada."
        elif "plano do vídeo de apresentação" in template:
            content = "## Plano da apresentação\nLeia e organize os dados sem resolver."
        else:
            content = "## Plano da resolução\nExplique a solução em etapas."
        return _service_result(content)

    def call_agent(
        self,
        request: Any,
        *,
        tools: object,
        response_schema: object = None,
    ) -> SimpleNamespace:
        _ = tools
        self.agent_requests.append(request)
        self.agent_response_schemas.append(response_schema)
        code = (
            _PRESENTATION_CODE
            if "vídeo de apresentação" in request.template_text
            else _SOLUTION_CODE
        )
        return _service_result(code)


class FakeRenderer:
    """Create deterministic video placeholders or configured failures."""

    def __init__(self, failures: int = 0) -> None:
        self.failures = failures
        self.calls = 0
        self.codes: list[str] = []

    def render(
        self, code: Any, *, project_directory: Path, mode: str, **kwargs: Any
    ) -> RenderResult:
        _ = kwargs
        self.calls += 1
        self.codes.append(code.code)
        if self.calls <= self.failures:
            return RenderResult(
                mode=mode,
                success=False,
                return_code=1,
                code_path=code.code_path,
                stderr="NameError: objeto não definido",
                attempts=1,
            )
        video_path = project_directory / mode / f"{mode}.mp4"
        video_path.parent.mkdir(parents=True, exist_ok=True)
        video_path.write_bytes(b"video")
        return RenderResult(
            mode=mode,
            success=True,
            return_code=0,
            video_path=str(video_path),
            code_path=code.code_path,
            attempts=1,
        )


class FakeVideoAssembler:
    """Create deterministic final-video placeholders without invoking FFmpeg."""

    def __init__(self, *, succeeds: bool = True) -> None:
        self.succeeds = succeeds
        self.calls = 0

    def combine(self, videos: tuple[Path, ...], output_path: Path) -> VideoAssemblyResult:
        self.calls += 1
        if not self.succeeds:
            return VideoAssemblyResult(success=False, error_message="FFmpeg indisponível")
        assert all(path.is_file() for path in videos)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"final-video")
        return VideoAssemblyResult(success=True, video_path=str(output_path))


def _service_result(content: str) -> SimpleNamespace:
    return SimpleNamespace(
        result=LLMCallResult(
            ok=True,
            provider="OpenAI",
            model="test-model",
            content=content,
        )
    )


def _workflow(
    tmp_path: Path,
    renderer: FakeRenderer,
    *,
    delivery_mode: str = "separate",
    video_assembler: FakeVideoAssembler | None = None,
    llm_service: FakeLLMService | None = None,
    teacher_solution: str = "## Solução do professor\nBase verificada.",
    solution_image: bool = False,
) -> tuple[Any, dict[str, Any]]:
    repository = ProjectRepository(tmp_path / "olympianim.db")
    project = repository.create_project(
        ProjectCreate(
            title="Caso",
            problem_statement="Determine o valor pedido.",
            problem_source="OBMEP",
            teacher_solution=teacher_solution,
            output_delivery_mode=delivery_mode,
        ),
        project_id="project-id",
    )
    if solution_image:
        image_path = tmp_path / "teacher_solution.png"
        image_path.write_bytes(b"png")
        repository.add_generated_file(
            project.id,
            file_type="solution_image",
            path=str(image_path),
            version=1,
            artifact_key="solution_image:v1",
        )
    graph = build_approved_workflow(
        llm_api_key="test-key",
        voice_api_key="",
        llm_service=llm_service or FakeLLMService(),
        prompt_service=PromptService(repository=repository),
        artifact_service=ArtifactService(
            repository=repository,
            projects_dir=tmp_path / "projects",
        ),
        renderer=renderer,
        repository=repository,
        project_id=project.id,
        checkpointer=MemorySaver(),
        video_assembler=video_assembler,
    )
    initial_state = {
        "project_id": project.id,
        "problem_statement": project.problem_statement,
        "teacher_solution": teacher_solution,
        "teacher_instructions": "",
        "llm_provider": "OpenAI",
        "llm_model": "test-model",
        "voice": {"enabled": False},
        "voice_prompt_template": "{transcript}",
        "retry_count": 0,
    }
    return graph, initial_state


def _config(name: str) -> dict[str, dict[str, str]]:
    return {"configurable": {"thread_id": name}}


def test_voice_contract_is_omitted_when_narration_is_disabled() -> None:
    assert voiceover_prompt_requirements(VoiceConfig(enabled=False)) == ""


def test_voice_contract_limits_builder_and_debugger_to_app_owned_api() -> None:
    requirements = voiceover_prompt_requirements(VoiceConfig(enabled=True))

    assert "from olympianim.manim.voiceover import ConfiguredVoiceoverScene" in requirements
    assert "não importe ``manim_voiceover``" in requirements
    assert "não pesquise essa camada" in requirements
    assert "somente com fala não vazia" in requirements
    assert "execute ``self.play(...)`` diretamente" in requirements
    assert "espera automaticamente o tempo de áudio restante" in requirements
    assert "nunca chame nem redefina ``wait_for_voiceover``" in requirements
    assert "não sobrescreva métodos herdados" in requirements
    assert "além de ``construct``" in requirements
    assert "nomes descritivos que não existam nas classes-base" in requirements
    assert "use ``tracker.duration`` apenas para dimensionar" in requirements
    assert "bloco curto e atômico" in requirements
    assert "contexto visual → fala → evento visual" in requirements
    assert "contexto de acompanhamento" in requirements
    assert "informação construída" in requirements
    assert "não prolongue o aparecimento desse contexto" in requirements
    assert "leia integralmente o enunciado" not in requirements
    assert "<bookmark mark='nome_unico'/>" in requirements
    assert "self.wait_until_bookmark('nome_unico')" in requirements
    assert "tracker.time_until_bookmark('proximo_marco')" in requirements
    assert "dados ou operandos" in requirements
    assert "nunca mostre uma igualdade final" in requirements


def test_debugger_voice_contract_omits_editorial_choreography() -> None:
    requirements = voiceover_prompt_requirements(
        VoiceConfig(enabled=True),
        synchronize_visuals=False,
    )

    assert "ConfiguredVoiceoverScene" in requirements
    assert "Sincronização didática obrigatória" not in requirements
    assert "bookmark" not in requirements


def test_complete_workflow_requires_all_human_approvals(tmp_path: Path) -> None:
    renderer = FakeRenderer()
    graph, initial_state = _workflow(tmp_path, renderer)
    config = _config("complete")

    graph.invoke(initial_state, config)
    assert graph.get_state(config).values["phase"] == "review_plan_presentation"

    graph.invoke(Command(resume={"action": "approve"}), config)
    assert graph.get_state(config).values["phase"] == "review_code_presentation"

    graph.invoke(Command(resume={"action": "approve"}), config)
    state = graph.get_state(config).values
    assert state["phase"] == "presentation_complete"
    assert Path(state["presentation_render_path"]).is_file()

    graph.invoke(Command(resume={"action": "generate_solution"}), config)
    assert graph.get_state(config).values["phase"] == "review_plan_solution"

    graph.invoke(Command(resume={"action": "approve"}), config)
    assert graph.get_state(config).values["phase"] == "review_code_solution"

    graph.invoke(Command(resume={"action": "approve"}), config)
    state = graph.get_state(config).values
    assert state["phase"] == "completed"
    assert Path(state["solution_render_path"]).is_file()
    assert renderer.calls == 2


def test_ai_solution_is_reviewed_before_presentation_and_reused(
    tmp_path: Path,
) -> None:
    llm_service = FakeLLMService()
    graph, initial_state = _workflow(
        tmp_path,
        FakeRenderer(),
        llm_service=llm_service,
        teacher_solution="",
    )
    config = _config("review-ai-solution")

    graph.invoke(initial_state, config)

    state = graph.get_state(config).values
    assert state["phase"] == "review_solution_basis"
    assert state["solution_basis_source"] == "ai_solution"
    assert state["solution_basis_approved"] is False
    assert len(llm_service.text_requests) == 1
    solver_request = llm_service.text_requests[0]
    assert "produza uma base matemática única" in solver_request.template_text.casefold()
    assert not solver_request.images
    assert "fonte vinculante do método" not in solver_request.template_text

    graph.invoke(
        Command(
            resume={
                "action": "approve",
                "model_selection": {"provider": "Google", "model": "planner-model"},
            }
        ),
        config,
    )

    state = graph.get_state(config).values
    assert state["phase"] == "review_plan_presentation"
    assert state["solution_basis_approved"] is True
    assert llm_service.text_requests[-1].prompt_values["solution_basis"] == state["solution_basis"]
    assert llm_service.text_requests[-1].provider == "Google"
    assert llm_service.text_requests[-1].model == "planner-model"

    graph.invoke(Command(resume={"action": "approve"}), config)
    graph.invoke(Command(resume={"action": "approve"}), config)
    graph.invoke(Command(resume={"action": "generate_solution"}), config)

    assert graph.get_state(config).values["phase"] == "review_plan_solution"
    solver_requests = [
        request
        for request in llm_service.text_requests
        if "produza uma base matemática única" in request.template_text.casefold()
    ]
    assert len(solver_requests) == 1


def test_solution_image_is_interpreted_and_reviewed_before_presentation(
    tmp_path: Path,
) -> None:
    llm_service = FakeLLMService()
    graph, initial_state = _workflow(
        tmp_path,
        FakeRenderer(),
        llm_service=llm_service,
        teacher_solution="",
        solution_image=True,
    )
    config = _config("review-image-solution")

    graph.invoke(initial_state, config)

    state = graph.get_state(config).values
    request = llm_service.text_requests[0]
    assert state["phase"] == "review_solution_basis"
    assert state["solution_basis_source"] == "teacher_image_interpretation"
    assert len(request.images) == 1
    assert "fonte vinculante do método" in request.template_text
    assert "não pode trocar o método por uma solução alternativa" in request.template_text

    graph.invoke(Command(resume={"action": "approve"}), config)

    presentation_request = llm_service.text_requests[-1]
    assert len(presentation_request.images) == 1
    assert "referência privada" in presentation_request.template_text


def test_teacher_text_skips_basis_review_and_is_private_from_presentation_builder(
    tmp_path: Path,
) -> None:
    llm_service = FakeLLMService()
    graph, initial_state = _workflow(
        tmp_path,
        FakeRenderer(),
        llm_service=llm_service,
        solution_image=True,
    )
    config = _config("teacher-basis")

    graph.invoke(initial_state, config)

    state = graph.get_state(config).values
    assert state["phase"] == "review_plan_presentation"
    assert state["solution_basis_source"] == "teacher_text"
    assert state["solution_basis_approved"] is True
    assert len(llm_service.text_requests) == 1
    assert len(llm_service.text_requests[0].images) == 1

    graph.invoke(Command(resume={"action": "approve"}), config)

    builder_request = llm_service.agent_requests[-1]
    assert builder_request.prompt_values.get("solution_basis") is None
    assert builder_request.images == ()
    assert llm_service.agent_response_schemas[-1] is ManimCodeOutput

    graph.invoke(Command(resume={"action": "approve"}), config)
    graph.invoke(Command(resume={"action": "generate_solution"}), config)
    graph.invoke(Command(resume={"action": "approve"}), config)

    solution_builder_request = llm_service.agent_requests[-1]
    assert len(solution_builder_request.images) == 1
    assert llm_service.agent_response_schemas[-1] is ManimCodeOutput


def test_solution_basis_edit_requires_separate_approval(tmp_path: Path) -> None:
    graph, initial_state = _workflow(
        tmp_path,
        FakeRenderer(),
        teacher_solution="",
    )
    config = _config("edit-solution-basis")
    graph.invoke(initial_state, config)

    graph.invoke(
        Command(resume={"action": "edit", "content": "## Base corrigida"}),
        config,
    )

    state = graph.get_state(config).values
    assert state["phase"] == "review_solution_basis"
    assert state["solution_basis"] == "## Base corrigida"
    assert state["solution_basis_approved"] is False

    graph.invoke(Command(resume={"action": "approve"}), config)
    assert graph.get_state(config).values["phase"] == "review_plan_presentation"


def test_solution_basis_regeneration_returns_to_unapproved_review(tmp_path: Path) -> None:
    llm_service = FakeLLMService()
    graph, initial_state = _workflow(
        tmp_path,
        FakeRenderer(),
        llm_service=llm_service,
        teacher_solution="",
    )
    config = _config("regenerate-solution-basis")
    graph.invoke(initial_state, config)
    first_basis = graph.get_state(config).values["solution_basis"]

    graph.invoke(Command(resume={"action": "regenerate"}), config)

    state = graph.get_state(config).values
    assert state["phase"] == "review_solution_basis"
    assert state["solution_basis"] == first_basis
    assert state["solution_basis_approved"] is False
    assert len(llm_service.text_requests) == 1


def test_model_selection_is_applied_atomically_when_interrupt_resumes(
    tmp_path: Path,
) -> None:
    llm_service = FakeLLMService()
    graph, initial_state = _workflow(
        tmp_path,
        FakeRenderer(),
        llm_service=llm_service,
    )
    config = _config("model-selection-resume")

    graph.invoke(initial_state, config)
    graph.invoke(
        Command(
            resume={
                "action": "approve",
                "model_selection": {"provider": "Google", "model": "builder-model"},
            }
        ),
        config,
    )

    state = graph.get_state(config).values
    builder_request = llm_service.agent_requests[-1]
    assert state["phase"] == "review_code_presentation"
    assert state["llm_provider"] == "Google"
    assert state["llm_model"] == "builder-model"
    assert builder_request.provider == "Google"
    assert builder_request.model == "builder-model"

    graph.invoke(Command(resume={"action": "approve"}), config)
    graph.invoke(
        Command(
            resume={
                "action": "generate_solution",
                "model_selection": {
                    "provider": "Anthropic",
                    "model": "solution-model",
                },
            }
        ),
        config,
    )

    state = graph.get_state(config).values
    solution_requests = llm_service.text_requests[-1:]
    assert state["phase"] == "review_plan_solution"
    assert state["llm_provider"] == "Anthropic"
    assert state["llm_model"] == "solution-model"
    assert all(request.provider == "Anthropic" for request in solution_requests)
    assert all(request.model == "solution-model" for request in solution_requests)


def test_workflow_keeps_watermark_out_of_canonical_agent_code(tmp_path: Path) -> None:
    renderer = FakeRenderer()
    graph, initial_state = _workflow(tmp_path, renderer)
    config = _config("canonical-code")

    graph.invoke(initial_state, config)
    graph.invoke(Command(resume={"action": "approve"}), config)
    graph.invoke(Command(resume={"action": "approve"}), config)

    state = graph.get_state(config).values
    saved = Path(state["code_path"]).read_text(encoding="utf-8")
    assert "OlympianimSourceWatermarkMixin" not in state["code"]
    assert renderer.codes[0] == saved
    assert saved.count("class OlympianimSourceWatermarkMixin:") == 1
    assert saved.count("PresentationScene(OlympianimSourceWatermarkMixin, Scene)") == 1


def test_debugger_receives_canonical_code_without_watermark(tmp_path: Path) -> None:
    llm_service = FakeLLMService()
    graph, initial_state = _workflow(
        tmp_path,
        FakeRenderer(failures=1),
        llm_service=llm_service,
    )
    config = _config("canonical-debugger")

    graph.invoke(initial_state, config)
    graph.invoke(Command(resume={"action": "approve"}), config)
    graph.invoke(Command(resume={"action": "approve"}), config)

    debugger_request = llm_service.agent_requests[-1]
    assert "OlympianimSourceWatermarkMixin" not in debugger_request.prompt_values["manim_code"]
    assert llm_service.agent_response_schemas[-1] is ManimCodeOutput


def test_workflow_records_rendered_prompt_snapshot(tmp_path: Path) -> None:
    graph, initial_state = _workflow(tmp_path, FakeRenderer())
    graph.invoke(initial_state, _config("prompt-snapshot"))

    with sqlite3.connect(tmp_path / "olympianim.db") as connection:
        rows = connection.execute("""
            SELECT p.agent_type, pp.rendered_prompt_snapshot
            FROM project_prompts AS pp
            JOIN prompts AS p ON p.id = pp.prompt_id
            WHERE pp.project_id = 'project-id'
            ORDER BY pp.created_at
            """).fetchall()

    assert rows
    assert rows[0][0] == "workflow_planner"
    assert "Determine o valor pedido." in rows[0][1]


def test_plan_edit_stays_at_review_and_invalidates_downstream_content(tmp_path: Path) -> None:
    graph, initial_state = _workflow(tmp_path, FakeRenderer())
    config = _config("edit")
    graph.invoke(initial_state, config)

    graph.invoke(
        Command(resume={"action": "edit", "content": "## Plano editado"}),
        config,
    )

    state = graph.get_state(config).values
    assert state["phase"] == "review_plan_presentation"
    assert state["plan"] == "## Plano editado"
    assert not state.get("code")
    assert not state.get("render_path")


def test_render_failure_is_repaired_automatically(tmp_path: Path) -> None:
    renderer = FakeRenderer(failures=1)
    graph, initial_state = _workflow(tmp_path, renderer)
    config = _config("repair")
    graph.invoke(initial_state, config)
    graph.invoke(Command(resume={"action": "approve"}), config)

    graph.invoke(Command(resume={"action": "approve"}), config)

    state = graph.get_state(config).values
    assert state["phase"] == "presentation_complete"
    assert state["retry_count"] == 1
    assert renderer.calls == 2


def test_invalid_builder_response_is_not_cached(tmp_path: Path) -> None:
    class InvalidBuilderLLM(FakeLLMService):
        def call_agent(
            self,
            request: Any,
            *,
            tools: object,
            response_schema: object = None,
        ) -> SimpleNamespace:
            _ = (request, tools, response_schema)
            return _service_result("from manim import (")

    graph, initial_state = _workflow(
        tmp_path,
        FakeRenderer(),
        llm_service=InvalidBuilderLLM(),
    )
    config = _config("invalid-builder-cache")
    graph.invoke(initial_state, config)

    with pytest.raises(RuntimeError, match="never closed"):
        graph.invoke(Command(resume={"action": "approve"}), config)

    cached_roles = {
        item.role
        for item in ProjectRepository(tmp_path / "olympianim.db").list_llm_call_cache("project-id")
    }
    assert "planner" in cached_roles
    assert "builder" not in cached_roles


def test_render_correction_stops_after_three_retries(tmp_path: Path) -> None:
    renderer = FakeRenderer(failures=10)
    graph, initial_state = _workflow(tmp_path, renderer)
    config = _config("failed")
    graph.invoke(initial_state, config)
    graph.invoke(Command(resume={"action": "approve"}), config)

    graph.invoke(Command(resume={"action": "approve"}), config)

    state = graph.get_state(config).values
    assert state["phase"] == "failed"
    assert state["retry_count"] == 3
    assert renderer.calls == 4


def test_registered_manual_render_can_restart_a_failed_presentation(
    tmp_path: Path,
) -> None:
    renderer = FakeRenderer(failures=10)
    graph, initial_state = _workflow(tmp_path, renderer)
    config = _config("manual-presentation-recovery")
    graph.invoke(initial_state, config)
    graph.invoke(Command(resume={"action": "approve"}), config)
    graph.invoke(Command(resume={"action": "approve"}), config)
    assert graph.get_state(config).values["phase"] == "failed"

    repository = ProjectRepository(tmp_path / "olympianim.db")
    artifacts = ArtifactService(
        repository=repository,
        projects_dir=tmp_path / "projects",
    )
    video_path = (
        artifacts.project_directory("project-id")
        / "presentation"
        / "versions"
        / "presentation_v7.mp4"
    )
    video_path.write_bytes(b"manual-video")
    artifacts.register_video(
        "project-id",
        mode="presentation",
        video_path=video_path,
        version=7,
    )

    graph.invoke(
        {
            "workflow_entry": MANUAL_PRESENTATION_RECOVERY_ENTRY,
            "project_id": "project-id",
            "mode": "presentation",
            "phase": "presentation_complete",
            "render_path": str(video_path),
            "presentation_render_path": str(video_path),
            "render_error": "",
            "retry_count": 0,
        },
        config,
    )

    recovered = graph.get_state(config)
    assert recovered.values["phase"] == "presentation_complete"
    assert any(task.interrupts for task in recovered.tasks)

    graph.invoke(Command(resume={"action": "generate_solution"}), config)

    state = graph.get_state(config).values
    assert state["phase"] == "review_plan_solution"
    assert state["presentation_render_path"] == str(video_path)
    assert state["render_error"] == ""
    assert state["workflow_entry"] == ""


def test_combined_delivery_assembles_final_video_after_solution(tmp_path: Path) -> None:
    assembler = FakeVideoAssembler()
    graph, initial_state = _workflow(
        tmp_path,
        FakeRenderer(),
        delivery_mode="combined",
        video_assembler=assembler,
    )
    config = _config("combined")

    graph.invoke(initial_state, config)
    graph.invoke(Command(resume={"action": "approve"}), config)
    graph.invoke(Command(resume={"action": "approve"}), config)
    graph.invoke(Command(resume={"action": "generate_solution"}), config)
    graph.invoke(Command(resume={"action": "approve"}), config)
    graph.invoke(Command(resume={"action": "approve"}), config)

    state = graph.get_state(config).values
    project = ProjectRepository(tmp_path / "olympianim.db").get_project("project-id")
    assert assembler.calls == 1
    assert Path(state["final_render_path"]).is_file()
    assert project is not None
    assert project.final_video_path == state["final_render_path"]


def test_combined_delivery_fails_without_removing_source_videos(tmp_path: Path) -> None:
    graph, initial_state = _workflow(
        tmp_path,
        FakeRenderer(),
        delivery_mode="combined",
        video_assembler=FakeVideoAssembler(succeeds=False),
    )
    config = _config("combined-failed")

    graph.invoke(initial_state, config)
    graph.invoke(Command(resume={"action": "approve"}), config)
    graph.invoke(Command(resume={"action": "approve"}), config)
    graph.invoke(Command(resume={"action": "generate_solution"}), config)
    graph.invoke(Command(resume={"action": "approve"}), config)
    graph.invoke(Command(resume={"action": "approve"}), config)

    state = graph.get_state(config).values
    assert state["phase"] == "failed"
    assert Path(state["presentation_render_path"]).is_file()
    assert Path(state["solution_render_path"]).is_file()
