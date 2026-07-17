"""Guided native LangGraph human-review screen."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import streamlit as st
from code_editor import code_editor

from olympianim.services.background_jobs import (
    BackgroundJobService,
    get_background_job_service,
    wake_background_worker,
)
from olympianim.services.langgraph_workflow import (
    LangGraphWorkflowService,
    WorkflowCredentials,
    WorkflowModelSelection,
)
from olympianim.services.project_service import ProjectService
from olympianim.services.subtitle_style import SubtitleStyle, SubtitleStyleService
from olympianim.services.subtitle_video_service import (
    SubtitleVideoMode,
    SubtitleVideoService,
    SubtitleVideoState,
)
from olympianim.ui import options, state
from olympianim.ui.credentials import workflow_credentials_for_project

_STEPS = (
    ("review_solution_basis", "1. Base matemática"),
    ("review_plan_presentation", "2. Plano da apresentação"),
    ("review_code_presentation", "3. Código da apresentação"),
    ("render_presentation", "4. Renderização da apresentação"),
    ("review_plan_solution", "5. Plano da resolução"),
    ("review_code_solution", "6. Código da resolução"),
    ("render_solution", "7. Renderização da resolução"),
)


def _run(action: Callable[[], object]) -> None:
    try:
        action()
    except Exception as exc:
        st.error(str(exc))
    else:
        st.rerun()


def _enqueue(
    service: BackgroundJobService,
    project_id: str,
    credentials: WorkflowCredentials,
    *,
    action: str,
    decision: dict[str, object] | None = None,
    model_selection: WorkflowModelSelection,
    expected_phase: str | None = None,
) -> None:
    """Queue one workflow transition and wake the local worker."""
    if not credentials.llm_api_key:
        raise ValueError(
            f"Nenhuma chave de API foi encontrada para {model_selection.provider}. "
            "Configure a chave desse provedor antes de continuar."
        )
    service.enqueue(
        project_id,
        action,
        decision=decision,
        credentials=credentials,
        model_selection=model_selection,
        expected_phase=expected_phase,
    )
    wake_background_worker()


@st.fragment(run_every=2.0)
def _render_active_job(service: BackgroundJobService, project_id: str) -> None:
    """Refresh persisted progress without rerunning the full page."""
    job = service.active_job(project_id)
    if job is None:
        st.rerun()
    assert job is not None
    label = "Na fila" if job.status == "pending" else "Processando"
    with st.status(f"{label}: {job.current_step}", state="running", expanded=True):
        st.progress(job.progress)
        st.caption(f"Tentativa {job.attempts or 1} · trabalho {job.id}")
        if job.cancel_requested:
            st.caption(
                "Cancelamento solicitado. A etapa será interrompida no próximo limite seguro."
            )
        elif st.button(
            "Cancelar etapa",
            key=f"cancel_job_{job.id}",
            icon=":material/stop:",
        ):
            service.cancel(job.id)
            st.rerun(scope="fragment")


def _available_video(*paths: object) -> str:
    """Return the first persisted video that is still available on disk."""
    for value in paths:
        path = str(value or "")
        if path and Path(path).is_file():
            return path
    return ""


def _render_steps(phase: str) -> None:
    current = next((index for index, (key, _) in enumerate(_STEPS) if key == phase), -1)
    completed_before = max(current, 0)
    if phase.startswith("repair_presentation"):
        current = next(
            index for index, (key, _) in enumerate(_STEPS) if key == "render_presentation"
        )
        completed_before = current
    elif phase.startswith("repair_solution"):
        current = next(index for index, (key, _) in enumerate(_STEPS) if key == "render_solution")
        completed_before = current
    elif phase == "completed":
        current = -1
        completed_before = len(_STEPS)
    elif phase == "presentation_complete":
        current = -1
        completed_before = next(
            index for index, (key, _) in enumerate(_STEPS) if key == "review_plan_solution"
        )
    for index, (_, label) in enumerate(_STEPS):
        if index < completed_before:
            status = "Concluída"
        elif index == current:
            status = "Em revisão"
        else:
            status = "Aguardando etapa anterior"
        st.caption(f"{label} · {status}")


def _render_video_with_accessibility(
    project_service: ProjectService,
    project_id: str,
    mode: SubtitleVideoMode,
    video_path: str,
) -> None:
    """Show the active MP4 and keep all subtitle actions in one compact area."""
    subtitle_service = SubtitleVideoService(
        repository=project_service.repository,
        projects_dir=project_service.projects_dir,
    )
    style_service = SubtitleStyleService(project_service.repository)
    stored_style = style_service.load(project_id, mode)
    color_key = f"subtitle_text_color_{project_id}_{mode}"
    st.session_state.setdefault(color_key, stored_style.text_color)
    try:
        selected_style = SubtitleStyle(str(st.session_state[color_key]))
    except ValueError:
        selected_style = stored_style
        st.session_state[color_key] = stored_style.text_color
    try:
        subtitle_state = subtitle_service.state(
            project_id,
            mode,
            video_path,
            style=selected_style,
        )
    except ValueError:
        st.video(video_path)
        return
    st.video(subtitle_state.display_video_path)
    _render_accessibility_controls(
        project_service,
        subtitle_service,
        project_id,
        mode,
        subtitle_state,
        style_service,
        stored_style,
        selected_style,
        color_key,
    )


def _render_accessibility_controls(
    project_service: ProjectService,
    subtitle_service: SubtitleVideoService,
    project_id: str,
    mode: SubtitleVideoMode,
    subtitle_state: SubtitleVideoState,
    style_service: SubtitleStyleService,
    stored_style: SubtitleStyle,
    selected_style: SubtitleStyle,
    color_key: str,
) -> None:
    """Expose downloads and reversible hard subtitles without cluttering the player."""
    records = project_service.repository.list_generated_files(project_id)
    files = {
        record.file_type: Path(record.path)
        for record in records
        if record.file_type in {f"{mode}_subtitle", f"{mode}_transcript"}
        and Path(record.path).is_file()
    }
    subtitle = files.get(f"{mode}_subtitle")
    transcript = files.get(f"{mode}_transcript")
    if subtitle is None and transcript is None:
        return
    with st.expander("Acessibilidade", expanded=False, icon=":material/accessibility_new:"):
        columns = st.columns(2)
        if subtitle is not None:
            columns[0].download_button(
                "Baixar legendas SRT",
                data=subtitle.read_bytes(),
                file_name=subtitle.name,
                mime="application/x-subrip",
                key=f"download_srt_{project_id}_{mode}",
                width="stretch",
            )
        if transcript is not None:
            columns[1].download_button(
                "Baixar transcrição",
                data=transcript.read_bytes(),
                file_name=transcript.name,
                mime="text/plain",
                key=f"download_transcript_{project_id}_{mode}",
                width="stretch",
            )
        if not subtitle_state.subtitle_available:
            return
        selected_color = st.color_picker(
            "Cor da legenda",
            key=color_key,
            help="A cor é aplicada somente à versão legendada; o vídeo original é preservado.",
            width="stretch",
        )
        if selected_color != stored_style.text_color:
            selected_style = style_service.save(project_id, mode, selected_color)
        if subtitle_state.captioned:
            if st.button(
                "Remover legendas do vídeo",
                key=f"remove_embedded_subtitles_{project_id}_{mode}",
                icon=":material/subtitles_off:",
                width="stretch",
            ):
                subtitle_service.remove(project_id, mode)
                st.toast("Legendas removidas. O vídeo original foi restaurado.")
                st.rerun()
            return
        if st.button(
            "Adicionar legendas ao vídeo",
            key=f"embed_subtitles_{project_id}_{mode}",
            icon=":material/closed_caption:",
            width="stretch",
        ):
            with st.spinner("Incorporando legendas ao vídeo..."):
                try:
                    result = subtitle_service.add(
                        project_id,
                        mode,
                        subtitle_state.original_video_path,
                        style=selected_style,
                    )
                except Exception as exc:
                    _render_subtitle_failure(str(exc))
                else:
                    if result.success:
                        st.toast("Legendas adicionadas ao vídeo.")
                        st.rerun()
                    else:
                        _render_subtitle_failure(result.error_message)


def _render_subtitle_failure(message: str) -> None:
    """Keep FFmpeg diagnostics available without overwhelming the main screen."""
    detail = message.strip() or "Falha sem diagnóstico disponível."
    st.error(detail.splitlines()[0][:240])
    with st.expander("Detalhes técnicos"):
        st.code(detail, language=None)


def _render_model_selector(
    project: object,
    *,
    disabled: bool,
) -> WorkflowModelSelection:
    """Render a project-scoped choice for the next AI transition."""
    project_id = str(getattr(project, "id", ""))
    provider_key = f"workflow_provider_{project_id}"
    model_key = f"workflow_model_{project_id}"
    providers = options.active_llm_providers()
    if not providers:
        st.error("Ative ao menos um modelo de texto em Configurações.")
        st.stop()

    project_provider = str(getattr(project, "llm_provider", ""))
    if st.session_state.get(provider_key) not in providers:
        st.session_state[provider_key] = (
            project_provider if project_provider in providers else providers[0]
        )

    with st.container(border=True):
        st.caption("Modelo para a próxima chamada de IA")
        provider_column, model_column = st.columns(2)
        with provider_column:
            provider = st.selectbox(
                "Provedor",
                providers,
                key=provider_key,
                disabled=disabled,
            )
        models = options.models_for(provider)
        if not models:
            st.error(f"Não há modelos de texto ativos para {provider}.")
            st.stop()
        project_model = str(getattr(project, "llm_model", ""))
        if st.session_state.get(model_key) not in models:
            st.session_state[model_key] = project_model if project_model in models else models[0]
        with model_column:
            model = st.selectbox(
                "Modelo",
                models,
                format_func=lambda value: options.model_label(provider, value),
                key=model_key,
                disabled=disabled,
            )
        st.caption(
            "A escolha é aplicada somente à próxima etapa e não altera o padrão do projeto."
        )
    return WorkflowModelSelection(provider=provider, model=model)


@st.dialog("Revisar plano", width="large")
def _open_plan_review(
    service: BackgroundJobService,
    project_id: str,
    mode: str,
    plan: str,
    credentials: WorkflowCredentials,
    model_selection: WorkflowModelSelection,
) -> None:
    with st.expander("Visualização Markdown", expanded=False):
        st.markdown(plan)
    content = str(
        code_editor(plan, lang="markdown", height=480, key=f"plan_{mode}").get("text") or plan
    )
    save, approve, regenerate, reject = st.columns(4)
    with save:
        if st.button("Salvar edição", icon=":material/save:", width="stretch"):
            _run(
                lambda: _enqueue(
                    service,
                    project_id,
                    credentials,
                    action="resume",
                    decision={"action": "edit", "content": content},
                    model_selection=model_selection,
                    expected_phase=f"review_plan_{mode}",
                )
            )
    with approve:
        if st.button(
            "Aprovar plano",
            type="primary",
            icon=":material/check:",
            width="stretch",
        ):
            _run(
                lambda: _enqueue(
                    service,
                    project_id,
                    credentials,
                    action="resume",
                    decision={"action": "approve"},
                    model_selection=model_selection,
                    expected_phase=f"review_plan_{mode}",
                )
            )
    with regenerate:
        if st.button("Gerar novamente", icon=":material/refresh:", width="stretch"):
            _run(
                lambda: _enqueue(
                    service,
                    project_id,
                    credentials,
                    action="resume",
                    decision={"action": "regenerate"},
                    model_selection=model_selection,
                    expected_phase=f"review_plan_{mode}",
                )
            )
    with reject:
        if st.button("Reprovar plano", icon=":material/close:", width="stretch"):
            _run(
                lambda: _enqueue(
                    service,
                    project_id,
                    credentials,
                    action="resume",
                    decision={"action": "reject"},
                    model_selection=model_selection,
                    expected_phase=f"review_plan_{mode}",
                )
            )


@st.dialog("Revisar base matemática", width="large")
def _open_solution_basis_review(
    service: BackgroundJobService,
    project_id: str,
    solution_basis: str,
    credentials: WorkflowCredentials,
    model_selection: WorkflowModelSelection,
) -> None:
    st.caption(
        "Confira a estratégia, os cálculos e a resposta. Depois de aprovada, esta base "
        "orientará a apresentação e a resolução."
    )
    with st.expander("Visualização Markdown", expanded=False):
        st.markdown(solution_basis)
    content = str(
        code_editor(
            solution_basis,
            lang="markdown",
            height=520,
            key=f"solution_basis_{project_id}",
        ).get("text")
        or solution_basis
    )
    save, approve, regenerate, reject = st.columns(4)
    with save:
        if st.button("Salvar edição", icon=":material/save:", width="stretch"):
            _run(
                lambda: _enqueue(
                    service,
                    project_id,
                    credentials,
                    action="resume",
                    decision={"action": "edit", "content": content},
                    model_selection=model_selection,
                    expected_phase="review_solution_basis",
                )
            )
    with approve:
        if st.button(
            "Aprovar base",
            type="primary",
            icon=":material/check:",
            width="stretch",
        ):
            _run(
                lambda: _enqueue(
                    service,
                    project_id,
                    credentials,
                    action="resume",
                    decision={"action": "approve"},
                    model_selection=model_selection,
                    expected_phase="review_solution_basis",
                )
            )
    with regenerate:
        if st.button("Gerar novamente", icon=":material/refresh:", width="stretch"):
            _run(
                lambda: _enqueue(
                    service,
                    project_id,
                    credentials,
                    action="resume",
                    decision={"action": "regenerate"},
                    model_selection=model_selection,
                    expected_phase="review_solution_basis",
                )
            )
    with reject:
        if st.button("Interromper", icon=":material/close:", width="stretch"):
            _run(
                lambda: _enqueue(
                    service,
                    project_id,
                    credentials,
                    action="resume",
                    decision={"action": "reject"},
                    model_selection=model_selection,
                    expected_phase="review_solution_basis",
                )
            )


@st.dialog("Revisar código Manim", width="large")
def _open_code_review(
    service: BackgroundJobService,
    project_id: str,
    mode: str,
    code: str,
    credentials: WorkflowCredentials,
    model_selection: WorkflowModelSelection,
) -> None:
    content = str(
        code_editor(code, lang="python", height=620, key=f"code_{mode}").get("text") or code
    )
    save, approve, regenerate = st.columns(3)
    with save:
        if st.button("Salvar edição", icon=":material/save:", width="stretch"):
            _run(
                lambda: _enqueue(
                    service,
                    project_id,
                    credentials,
                    action="resume",
                    decision={"action": "edit", "content": content},
                    model_selection=model_selection,
                    expected_phase=f"review_code_{mode}",
                )
            )
    with approve:
        if st.button(
            "Aprovar código e renderizar",
            type="primary",
            icon=":material/play_arrow:",
            width="stretch",
        ):
            _run(
                lambda: _enqueue(
                    service,
                    project_id,
                    credentials,
                    action="resume",
                    decision={"action": "approve"},
                    model_selection=model_selection,
                    expected_phase=f"review_code_{mode}",
                )
            )
    with regenerate:
        if st.button("Gerar novamente", icon=":material/refresh:", width="stretch"):
            _run(
                lambda: _enqueue(
                    service,
                    project_id,
                    credentials,
                    action="resume",
                    decision={"action": "regenerate"},
                    model_selection=model_selection,
                    expected_phase=f"review_code_{mode}",
                )
            )


st.title("Produção de vídeos")
project_id = str(state.get(state.KEY_CURRENT_PROJECT_ID, ""))
if not project_id:
    st.info("Crie um projeto na página Início ou abra um projeto salvo.")
    st.stop()

project_service = ProjectService()
project = project_service.open_project(project_id)
if project is None:
    st.error("O projeto selecionado não existe mais.")
    st.stop()

workflow_service = LangGraphWorkflowService()
job_service = get_background_job_service()
st.caption(project.title)
animation_assets = project_service.list_animation_assets(project_id)
if animation_assets:
    with st.expander("Objetos visuais do projeto", expanded=False):
        columns = st.columns(min(len(animation_assets), 3))
        for index, asset in enumerate(animation_assets):
            with columns[index % len(columns)]:
                st.image(asset.path, caption=asset.filename, width="stretch")
                st.caption(asset.description)
with st.expander("Log do projeto", expanded=False):
    logs = project_service.list_logs(project_id)
    if not logs:
        st.caption("Nenhum evento registrado.")
    for entry in reversed(logs[-100:]):
        st.caption(f"{entry.created_at} · {entry.level.upper()} · {entry.step}")
        st.write(entry.message)

active_job = job_service.active_job(project_id)
model_selection = _render_model_selector(project, disabled=active_job is not None)
credentials = workflow_credentials_for_project(project, model_selection.provider)
snapshot = workflow_service.snapshot(project_id, credentials=credentials)
if active_job is not None:
    _render_active_job(job_service, project_id)
    st.stop()

recoverable_presentation = workflow_service.recoverable_manual_presentation_video(
    project_id,
    snapshot,
)
latest_job = job_service.latest_job(project_id)
if (
    latest_job is not None
    and latest_job.status in {"failed", "cancelled"}
    and project.status in {"failed", "cancelled"}
    and not recoverable_presentation
):
    if latest_job.status == "failed":
        st.error(latest_job.error_message or "A etapa em segundo plano falhou.")
        label = "Repetir etapa"
    else:
        st.warning("A última etapa foi cancelada.")
        label = "Retomar etapa"
    terminal_job_id = latest_job.id
    if st.button(
        label,
        type="primary",
        key=f"retry_job_{terminal_job_id}",
        icon=":material/replay:",
    ):
        _run(
            lambda: job_service.retry_and_wake(
                terminal_job_id,
                credentials=credentials,
                model_selection=model_selection,
            )
        )
    st.stop()

if not snapshot:
    st.subheader("1. Base matemática")
    st.caption(
        "A produção começa consolidando a solução que orientará os dois vídeos. "
        "Bases produzidas pela IA serão apresentadas para sua aprovação."
    )
    if st.button(
        "Iniciar produção",
        type="primary",
        icon=":material/auto_awesome:",
    ):
        _run(
            lambda: _enqueue(
                job_service,
                project_id,
                credentials,
                action="start",
                model_selection=model_selection,
            )
        )
    st.stop()

checkpoint_phase = str(snapshot.get("phase", ""))
phase = "presentation_complete" if recoverable_presentation else checkpoint_phase
mode = str(snapshot.get("mode", "presentation"))
with st.container(border=True):
    _render_steps(phase)
st.divider()

if phase == "review_solution_basis":
    st.subheader("Base matemática")
    source = str(snapshot.get("solution_basis_source", "ai_solution"))
    if source == "teacher_image_interpretation":
        st.caption(
            "Revise a interpretação feita a partir da solução anexada antes de planejar os vídeos."
        )
    else:
        st.caption(
            "Revise a solução produzida pelo agente antes de usá-la como referência didática."
        )
    if st.button(
        "Abrir base para revisar",
        type="primary",
        icon=":material/rate_review:",
    ):
        _open_solution_basis_review(
            job_service,
            project_id,
            str(snapshot["solution_basis"]),
            credentials,
            model_selection,
        )
elif phase.startswith("review_plan_"):
    title = "Plano da apresentação" if mode == "presentation" else "Plano da resolução"
    st.subheader(title)
    st.caption("Revise, ajuste ou aprove o roteiro para liberar a geração do código.")
    if st.button(
        "Abrir plano para revisar",
        type="primary",
        icon=":material/rate_review:",
    ):
        _open_plan_review(
            job_service,
            project_id,
            mode,
            str(snapshot["plan"]),
            credentials,
            model_selection,
        )
elif phase.startswith("review_code_"):
    title = "Código da apresentação" if mode == "presentation" else "Código da resolução"
    st.subheader(title)
    st.caption("Revise o código antes da renderização. Nenhum vídeo será gerado sem aprovação.")
    if st.button(
        "Abrir código para revisar",
        type="primary",
        icon=":material/code:",
    ):
        _open_code_review(
            job_service,
            project_id,
            mode,
            str(snapshot["code"]),
            credentials,
            model_selection,
        )
elif phase == "presentation_complete":
    st.subheader("Vídeo de apresentação")
    presentation_video = _available_video(
        recoverable_presentation,
        snapshot.get("presentation_render_path"),
        project.presentation_video_path,
        snapshot.get("render_path"),
    )
    if presentation_video:
        _render_video_with_accessibility(
            project_service,
            project_id,
            "presentation",
            presentation_video,
        )
    else:
        st.warning("O arquivo do vídeo de apresentação não foi encontrado.")
    if recoverable_presentation:
        st.caption(
            "A apresentação renderizada no Editor Manim está disponível. "
            "Ao gerar a resolução, o fluxo será retomado a partir deste vídeo."
        )
    else:
        st.caption(
            "A apresentação foi concluída. "
            "A resolução será iniciada somente quando você solicitar."
        )
    if st.button(
        "Gerar resolução",
        type="primary",
        icon=":material/auto_awesome:",
        key=f"generate_solution_{project_id}",
    ):
        _run(
            lambda: _enqueue(
                job_service,
                project_id,
                credentials,
                action="resume",
                decision={"action": "generate_solution"},
                model_selection=model_selection,
                expected_phase="presentation_complete",
            )
        )
elif phase.startswith(("prepare_", "plan_", "build_", "repair_", "render_")):
    _run(
        lambda: _enqueue(
            job_service,
            project_id,
            credentials,
            action="continue",
            model_selection=model_selection,
        )
    )
elif phase == "completed":
    st.subheader("Produção concluída")
    presentation_video = _available_video(
        snapshot.get("presentation_render_path"),
        project.presentation_video_path,
    )
    solution_video = _available_video(
        snapshot.get("solution_render_path"),
        project.solution_video_path,
        snapshot.get("render_path"),
    )
    final_video = _available_video(
        snapshot.get("final_render_path"),
        project.final_video_path,
    )
    tab_labels = ["Apresentação", "Resolução"]
    if final_video:
        tab_labels.insert(0, "Vídeo único")
    tabs = st.tabs(tab_labels)
    if final_video:
        with tabs[0]:
            _render_video_with_accessibility(
                project_service,
                project_id,
                "final",
                final_video,
            )
        presentation_tab, solution_tab = tabs[1:]
    else:
        presentation_tab, solution_tab = tabs
    with presentation_tab:
        if presentation_video:
            _render_video_with_accessibility(
                project_service,
                project_id,
                "presentation",
                presentation_video,
            )
        else:
            st.warning("O arquivo do vídeo de apresentação não foi encontrado.")
    with solution_tab:
        if solution_video:
            _render_video_with_accessibility(
                project_service,
                project_id,
                "solution",
                solution_video,
            )
        else:
            st.warning("O arquivo do vídeo de resolução não foi encontrado.")
elif phase == "failed":
    render_error = str(
        snapshot.get("render_error", "A renderização não pôde ser concluída.")
    ).strip()
    st.error((render_error.splitlines()[0] if render_error else "Falha sem diagnóstico.")[:240])
    if render_error:
        with st.expander("Detalhes técnicos do erro"):
            st.code(render_error, language=None)
    presentation_video = _available_video(
        snapshot.get("presentation_render_path"),
        project.presentation_video_path,
    )
    solution_video = _available_video(
        snapshot.get("solution_render_path"),
        project.solution_video_path,
    )
    if presentation_video or solution_video:
        st.caption("Os vídeos individuais já concluídos foram preservados.")
        presentation_tab, solution_tab = st.tabs(["Apresentação", "Resolução"])
        with presentation_tab:
            if presentation_video:
                _render_video_with_accessibility(
                    project_service,
                    project_id,
                    "presentation",
                    presentation_video,
                )
            else:
                st.info("A apresentação ainda não foi renderizada.")
        with solution_tab:
            if solution_video:
                _render_video_with_accessibility(
                    project_service,
                    project_id,
                    "solution",
                    solution_video,
                )
            else:
                st.info("A resolução ainda não foi renderizada.")
    st.caption("As correções automáticas terminaram. Edite o código preservado manualmente.")
    if st.button(
        "Abrir no Editor Manim",
        type="primary",
        icon=":material/code:",
    ):
        st.switch_page("pages/code_editor.py")
else:
    st.status("Processando a próxima etapa", state="running")
