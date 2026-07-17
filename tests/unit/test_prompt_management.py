"""Tests for the active prompt catalog and editable prompt service."""

from __future__ import annotations

import hashlib
from pathlib import Path

from olympianim.database.models import ProjectCreate
from olympianim.database.repository import ProjectRepository
from olympianim.prompts.defaults import DEFAULT_PROMPTS, WORKFLOW_PROMPT_NAMES
from olympianim.prompts.service import prompt_service_for_database
from olympianim.prompts.validator import (
    extract_template_variables,
    render_prompt_template,
    validate_prompt_template,
)
from olympianim.prompts.variables import AGENT_SPECS


def test_default_prompts_cover_only_active_agents() -> None:
    assert {prompt.agent_type for prompt in DEFAULT_PROMPTS} == {
        agent.agent_type for agent in AGENT_SPECS
    }
    assert len(DEFAULT_PROMPTS) == 9
    assert set(WORKFLOW_PROMPT_NAMES) == {
        ("planner", "presentation"),
        ("planner", "solution"),
        ("builder", "presentation"),
        ("builder", "solution"),
        ("solver", "solution"),
        ("debugger", "presentation"),
        ("debugger", "solution"),
    }


def test_default_prompt_variables_are_valid() -> None:
    for prompt in DEFAULT_PROMPTS:
        assert validate_prompt_template(prompt.agent_type, prompt.template_text).valid


def test_default_prompts_render_without_pending_placeholders() -> None:
    values = {
        "problem_statement": "Enunciado",
        "teacher_instructions": "Instruções",
        "solution_basis": "Solução",
        "approved_plan": "Plano",
        "voiceover_requirements": "",
        "manim_code": "from manim import Scene",
        "render_error": "Erro",
        "transcript": "Narração",
        "language": "pt-BR",
        "video_mode": "apresentação",
    }

    for prompt in DEFAULT_PROMPTS:
        rendered = render_prompt_template(prompt.template_text, values)
        assert extract_template_variables(rendered) == ()


def test_prompts_use_specific_professional_identities() -> None:
    legacy_openings = (
        "Você é o único agente planejador reutilizável",
        "Você é o único agente builder reutilizável",
    )

    assert all(
        legacy not in prompt.template_text
        for prompt in DEFAULT_PROMPTS
        for legacy in legacy_openings
    )
    assert all("# Identidade" in prompt.template_text for prompt in DEFAULT_PROMPTS)


def test_only_planners_own_polya_and_pedagogical_sequence() -> None:
    planners = [prompt for prompt in DEFAULT_PROMPTS if prompt.agent_type == "workflow_planner"]
    other_agents = [
        prompt for prompt in DEFAULT_PROMPTS if prompt.agent_type != "workflow_planner"
    ]

    assert len(planners) == 2
    assert all("Pólya" in prompt.template_text for prompt in planners)
    assert all("Pólya" not in prompt.template_text for prompt in other_agents)

    presentation = next(prompt for prompt in planners if "apresentação" in prompt.name)
    solution = next(prompt for prompt in planners if "resolução" in prompt.name)
    assert "plano curto" in presentation.template_text
    assert "compreensão do problema" in presentation.template_text
    assert "progressão do geral para o particular" in presentation.template_text
    assert "preserve o esforço produtivo" in presentation.template_text
    assert "referência privada da solução" in presentation.template_text
    assert "formule perguntas a partir dos pré-requisitos" in presentation.template_text
    assert "resultado intermediário que determine sozinho o caminho" in (
        presentation.template_text
    )
    assert "verifique se o plano ainda permite ao aluno" in presentation.template_text
    assert "<referencia_privada_da_solucao>" in presentation.template_text
    assert "contexto visual → fala → evento visual" in presentation.template_text
    assert "antes da primeira palavra" in presentation.template_text
    assert "contexto de acompanhamento" in presentation.template_text
    assert "informação construída" in presentation.template_text
    assert "recapitulação" in presentation.template_text
    assert "Agora é sua vez de tentar resolver" in presentation.template_text
    assert "deve ser a última fala" in presentation.template_text
    assert "evite perguntas genéricas" in presentation.template_text
    assert "pergunta → pausa com tela estável → pista visual mínima" in presentation.template_text
    assert "não execute, complete nem revele visualmente" in presentation.template_text
    assert "não mencione Pólya" in presentation.template_text
    assert (
        "elaborar o plano, executar o plano e olhar retrospectivamente" in solution.template_text
    )
    assert "cada cena avançar o raciocínio" in solution.template_text
    assert "contexto visual → fala → evento visual" in solution.template_text
    assert "Agora, vamos à resolução." in solution.template_text
    assert "não o leia, não o parafraseie" in solution.template_text
    assert "figura-base, objetos comparados" in solution.template_text
    assert "dados ou operandos → operação ou transformação → resultado" in (solution.template_text)
    assert "omita métodos alternativos" in solution.template_text


def test_mathematical_agents_require_precise_language_and_valid_arguments() -> None:
    planners = [prompt for prompt in DEFAULT_PROMPTS if prompt.agent_type == "workflow_planner"]
    solver = next(prompt for prompt in DEFAULT_PROMPTS if prompt.agent_type == "solution_solver")
    presentation = next(prompt for prompt in planners if "apresentação" in prompt.name)
    solution = next(prompt for prompt in planners if "resolução" in prompt.name)

    mathematical_agents = (*planners, solver)
    assert all(
        "terminologia matemática consagrada" in prompt.template_text
        for prompt in mathematical_agents
    )
    assert all(
        "defina todo símbolo novo" in prompt.template_text for prompt in mathematical_agents
    )
    assert all(
        "nunca como prova de uma afirmação geral" in prompt.template_text
        for prompt in mathematical_agents
    )

    proof_agents = (solution, solver)
    assert all(
        "hipóteses disponíveis → inferência justificada → conclusão" in prompt.template_text
        for prompt in proof_agents
    )
    assert all(
        "condição necessária de suficiente" in prompt.template_text for prompt in proof_agents
    )
    assert all(
        "domínio, divisor não nulo, sinais, reversibilidade" in prompt.template_text
        for prompt in proof_agents
    )
    assert all("não suponha a recíproca" in prompt.template_text for prompt in proof_agents)
    assert "# Validade do argumento" not in presentation.template_text


def test_builders_treat_approved_plan_as_editorial_specification() -> None:
    builders = [prompt for prompt in DEFAULT_PROMPTS if prompt.agent_type == "workflow_builder"]

    assert len(builders) == 2
    assert all("especificação editorial definitiva" in prompt.template_text for prompt in builders)
    assert all(
        "sem acrescentar decisões pedagógicas" in prompt.template_text for prompt in builders
    )
    assert all("plano aprovado" in prompt.template_text for prompt in builders)
    assert all(
        "contexto visual → fala → evento visual" in prompt.template_text for prompt in builders
    )
    assert all("não una falas pertencentes" in prompt.template_text for prompt in builders)
    assert all("contexto de acompanhamento" in prompt.template_text for prompt in builders)
    assert all("não acrescente" in prompt.template_text for prompt in builders)
    presentation = next(prompt for prompt in builders if "apresentação" in prompt.name)
    solution = next(prompt for prompt in builders if "resolução" in prompt.name)
    assert "antes da narração inicial" in presentation.template_text
    assert "sem ler, parafrasear ou repetir" in solution.template_text
    assert "Agora, vamos à resolução." in solution.template_text
    assert "não escreva de uma vez uma expressão" in solution.template_text


def test_code_producers_require_code_only_responses() -> None:
    direct_code_prompts = [
        prompt
        for prompt in DEFAULT_PROMPTS
        if prompt.agent_type in {"workflow_builder", "workflow_debugger"}
    ]
    editor = next(
        prompt for prompt in DEFAULT_PROMPTS if prompt.name == "Editor Manim com IA - padrão"
    )

    assert all("campo ``code``" in prompt.template_text for prompt in direct_code_prompts)
    assert all("sem cercas Markdown" in prompt.template_text for prompt in direct_code_prompts)
    assert all(
        "introdução, resumo, explicação" in prompt.template_text for prompt in direct_code_prompts
    )
    assert "no campo ``code``" in editor.template_text
    assert "sem cercas Markdown nem explicações" in editor.template_text
    assert all("``np.sqrt(x)``" not in prompt.template_text for prompt in DEFAULT_PROMPTS)


def test_solver_and_debugger_respect_their_boundaries() -> None:
    solver = next(prompt for prompt in DEFAULT_PROMPTS if prompt.agent_type == "solution_solver")
    debugger = next(
        prompt for prompt in DEFAULT_PROMPTS if prompt.agent_type == "workflow_debugger"
    )

    assert "não planeje cenas, animações, layout ou narração" in solver.template_text
    assert "imagens da solução do professor" not in solver.template_text
    assert "fonte vinculante" not in solver.template_text
    assert "solução alternativa" not in solver.template_text
    assert "preserve literalmente cenas, layout, textos" in debugger.template_text
    assert "corrija somente a causa técnica" in debugger.template_text
    assert "não faça melhorias estéticas" in debugger.template_text


def test_builders_and_debugger_request_official_manim_reference() -> None:
    prompts = [
        prompt
        for prompt in DEFAULT_PROMPTS
        if prompt.agent_type in {"workflow_builder", "workflow_debugger"}
    ]
    assert all("search_manim_reference" in prompt.template_text for prompt in prompts)


def test_builder_prompts_use_a_compact_relative_layout_contract() -> None:
    builders = [prompt for prompt in DEFAULT_PROMPTS if prompt.agent_type == "workflow_builder"]

    assert all(
        "uma ideia visual principal por tela" in prompt.template_text for prompt in builders
    )
    assert all("``arrange``/``next_to``" in prompt.template_text for prompt in builders)
    assert all(
        "remova ou substitua a tela anterior" in prompt.template_text for prompt in builders
    )
    assert all("divida-a em telas consecutivas" in prompt.template_text for prompt in builders)


def test_builder_prompts_avoid_fragile_layout_heuristics() -> None:
    builders = [prompt for prompt in DEFAULT_PROMPTS if prompt.agent_type == "workflow_builder"]

    assert all("uma linha lógica por ``Tex``" in prompt.template_text for prompt in builders)
    assert all("fit_to_frame" not in prompt.template_text for prompt in builders)
    assert all("11.5 por 6.2" not in prompt.template_text for prompt in builders)
    assert all("índices de caracteres" not in prompt.template_text for prompt in builders)
    assert all("geometria final do alvo" in prompt.template_text for prompt in builders)


def test_builders_choose_text_objects_by_mathematical_content() -> None:
    prompts = [prompt for prompt in DEFAULT_PROMPTS if prompt.agent_type == "workflow_builder"]

    assert all(
        "somente quando o bloco não contiver notação matemática" in prompt.template_text
        for prompt in prompts
    )
    assert all(
        "use ``Tex`` em modo normal para a linha lógica completa" in prompt.template_text
        for prompt in prompts
    )
    assert all(
        "use ``MathTex`` quando o objeto inteiro for uma fórmula" in prompt.template_text
        for prompt in prompts
    )
    assert all("statement_lines = (" in prompt.template_text for prompt in prompts)
    assert all("*(Tex(line" in prompt.template_text for prompt in prompts)
    assert all(
        "não coloque uma frase inteira em ``MathTex``" in prompt.template_text
        for prompt in prompts
    )
    assert all(
        "o exemplo abaixo é apenas um padrão estrutural" in prompt.template_text
        for prompt in prompts
    )
    assert all('Text("Se"' not in prompt.template_text for prompt in prompts)
    assert all(
        "se já houver um ``ImageMobject``, reúna-o com ``Group``" in prompt.template_text
        for prompt in prompts
    )
    assert all("componentes pertencem ao Olympianim" in prompt.template_text for prompt in prompts)


def test_debugger_is_strictly_technical_and_conservative() -> None:
    debugger = next(
        prompt for prompt in DEFAULT_PROMPTS if prompt.agent_type == "workflow_debugger"
    )

    assert "assinatura de API" in debugger.template_text
    assert "reorganizações preventivas" in debugger.template_text
    assert "problemas técnicos ou visuais correlatos" not in debugger.template_text
    assert "risco de colisão" not in debugger.template_text


def test_prompt_service_seeds_current_defaults_once(tmp_path: Path) -> None:
    service = prompt_service_for_database(tmp_path / "olympianim.db")
    service.ensure_default_prompts()
    service.ensure_default_prompts()

    prompts = service.list_prompts()
    assert len(prompts) == len(DEFAULT_PROMPTS)
    assert all(item.latest_version.version == 1 for item in prompts)


def test_prompt_service_preserves_teacher_edit(tmp_path: Path) -> None:
    service = prompt_service_for_database(tmp_path / "olympianim.db")
    prompt = service.list_prompts("workflow_builder")[0]
    edited = prompt.latest_version.template_text + "\nRegra personalizada."

    version, validation = service.save_prompt_version(prompt.prompt.id, edited)

    assert validation.valid
    assert version is not None
    service.ensure_default_prompts()
    current = service.get_prompt(prompt.prompt.id)
    assert current is not None
    assert current.latest_version.template_text == edited


def test_code_assistant_defaults_restore_their_own_templates(tmp_path: Path) -> None:
    service = prompt_service_for_database(tmp_path / "olympianim.db")
    prompts = service.list_prompts("code_editor_agent")
    assert {item.prompt.name for item in prompts} == {
        "Conversa sobre código Manim - padrão",
        "Editor Manim com IA - padrão",
    }

    for prompt in prompts:
        duplicate = service.duplicate_prompt(
            prompt.prompt.id,
            f"{prompt.prompt.name} - copia",
        )
        service.save_prompt_version(duplicate.prompt.id, "{manim_code}")
        restored = service.restore_default_prompt(duplicate.prompt.id)

        assert restored.template_text == prompt.latest_version.template_text


def test_code_editor_preserves_atomic_narration_choreography() -> None:
    editor = next(
        prompt for prompt in DEFAULT_PROMPTS if prompt.name == "Editor Manim com IA - padrão"
    )

    assert "ao alterar fala, ordem ou animação" in editor.template_text
    assert "unidades curtas ``contexto visual → fala → evento visual``" in editor.template_text
    assert "objeto-base que o aluno precisa observar" in editor.template_text
    assert "na apresentação, o enunciado fica completo antes de ser lido" in (editor.template_text)
    assert "na resolução" in editor.template_text
    assert "destaques, transformações, inferências, operações e resultados" in (
        editor.template_text
    )


def test_prompt_service_upgrades_untouched_defaults_for_every_agent(tmp_path: Path) -> None:
    repository = ProjectRepository(tmp_path / "olympianim.db")
    for default in DEFAULT_PROMPTS:
        prompt = repository.create_prompt(
            name=default.name,
            agent_type=default.agent_type,
            description=default.description,
            is_default=True,
        )
        previous = "Template padrão anterior."
        repository.add_prompt_version(prompt.id, template_text=previous)
        repository.set_setting(
            f"prompt.default_sha256.{prompt.id}",
            hashlib.sha256(previous.encode()).hexdigest(),
        )

    service = prompt_service_for_database(tmp_path / "olympianim.db")
    service.ensure_default_prompts()

    upgraded = service.list_prompts()
    expected = {
        (prompt.agent_type, prompt.name): prompt.template_text for prompt in DEFAULT_PROMPTS
    }
    assert all(item.latest_version.version == 2 for item in upgraded)
    assert all(
        item.latest_version.template_text == expected[(item.prompt.agent_type, item.prompt.name)]
        for item in upgraded
    )


def test_prompt_service_does_not_guess_that_unknown_text_is_a_builtin(tmp_path: Path) -> None:
    repository = ProjectRepository(tmp_path / "olympianim.db")
    default = DEFAULT_PROMPTS[0]
    prompt = repository.create_prompt(
        name=default.name,
        agent_type=default.agent_type,
        description=default.description,
        is_default=True,
    )
    repository.add_prompt_version(prompt.id, template_text="Edição do professor.")

    prompt_service_for_database(tmp_path / "olympianim.db").ensure_default_prompts()

    latest = repository.get_latest_prompt_version(prompt.id)
    assert latest is not None
    assert latest.version == 1
    assert latest.template_text == "Edição do professor."


def test_prompt_service_records_project_snapshot(tmp_path: Path) -> None:
    database_path = tmp_path / "olympianim.db"
    repository = ProjectRepository(database_path)
    project = repository.create_project(
        ProjectCreate(title="Projeto", problem_statement="Problema")
    )
    service = prompt_service_for_database(database_path)
    prompt = service.list_prompts("workflow_planner")[0]

    snapshot = service.save_project_prompt_snapshot(
        project.id,
        agent_type=prompt.prompt.agent_type,
        prompt_id=prompt.prompt.id,
        values={"problem_statement": "Enunciado", "teacher_instructions": "Seja claro"},
    )

    assert snapshot.project_id == project.id
    assert "Enunciado" in snapshot.rendered_prompt_snapshot
