"""Projects page: list and reopen previously saved projects."""

from __future__ import annotations

import re

import streamlit as st

from olympianim.services.project_export import ProjectExportService
from olympianim.services.project_service import ProjectService
from olympianim.ui import state


def _load_project_into_session(project_id: str) -> None:
    service = ProjectService()
    project = service.open_project(project_id)
    if project is None:
        st.error("Projeto não encontrado.")
        return

    state.select_project(project)
    st.success("Projeto selecionado para continuar o fluxo de geração.")


def _render_export(project: object, service: ProjectExportService) -> None:
    """Offer a compact, rights-aware evidence export for one project."""
    project_id = str(getattr(project, "id", ""))
    title = str(getattr(project, "title", "projeto"))
    with st.popover("Exportar", icon=":material/archive:", width="stretch"):
        st.caption(
            "Inclui manifesto, prompts, eventos, consumo, códigos, diffs, hashes e legendas."
        )
        include_originals = st.checkbox(
            "Incluir imagens e vídeos originais",
            key=f"export_originals_{project_id}",
            help="Ficam fora do pacote por padrão.",
        )
        rights_confirmed = False
        if include_originals:
            rights_confirmed = st.checkbox(
                "Confirmo que posso compartilhar esses arquivos",
                key=f"export_rights_{project_id}",
            )
        cache_key = f"project_export_{project_id}"
        selection = (include_originals, rights_confirmed)
        cached = st.session_state.get(cache_key)
        if st.button(
            "Preparar pacote",
            key=f"prepare_export_{project_id}",
            disabled=include_originals and not rights_confirmed,
            width="stretch",
        ):
            try:
                payload = service.build_zip(
                    project_id,
                    include_originals=include_originals,
                    rights_confirmed=rights_confirmed,
                )
            except Exception as exc:
                st.error(str(exc))
            else:
                st.session_state[cache_key] = (selection, payload)
                cached = (selection, payload)
        if cached and cached[0] == selection:
            safe_title = re.sub(r"[^A-Za-z0-9_-]+", "_", title).strip("_") or "projeto"
            st.download_button(
                "Baixar ZIP auditável",
                data=cached[1],
                file_name=f"olympianim_{safe_title}.zip",
                mime="application/zip",
                key=f"download_export_{project_id}",
                type="primary",
                width="stretch",
            )


st.title("Projetos")
st.caption("Listagem e reabertura de projetos salvos no SQLite local.")

service = ProjectService()
export_service = ProjectExportService(repository=service.repository)
projects = service.list_projects()

if not projects:
    st.info("Nenhum projeto salvo ainda. Crie um projeto na página inicial.")
else:
    for project in projects:
        with st.container(border=True):
            col_info, col_action = st.columns([4, 1])
            with col_info:
                st.markdown(f"**{project.title}**")
                st.caption(
                    f"Status: {project.status} | IA: {project.llm_provider} "
                    f"{project.llm_model} | Atualizado em {project.updated_at}"
                )
                st.code(project.id, language=None)
            with col_action:
                if st.button(
                    "Abrir",
                    key=f"open_project_{project.id}",
                    icon=":material/folder_open:",
                    width="stretch",
                ):
                    _load_project_into_session(project.id)
                    st.switch_page("pages/generate_videos.py")
                _render_export(project, export_service)
