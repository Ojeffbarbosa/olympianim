"""Project creation actions for the Streamlit home page."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import streamlit as st

from olympianim.services.image_asset_service import MAX_ANIMATION_ASSETS
from olympianim.ui import state


def render_generate_button(form_data: Mapping[str, Any]) -> bool:
    """Validate the home form and persist a project when submitted."""
    pressed = st.form_submit_button(
        "Criar projeto e iniciar fluxo",
        type="primary",
        icon=":material/add:",
        width="stretch",
    )
    if not pressed:
        return False
    if not str(form_data.get("project_title", "")).strip():
        st.error("Informe um nome para o projeto.")
        return False
    if not str(form_data.get("statement", "")).strip() and not form_data.get("images"):
        st.error("Digite o enunciado ou envie ao menos uma imagem da questão.")
        return False
    if int(form_data.get("animation_asset_upload_count", 0)) > MAX_ANIMATION_ASSETS:
        st.error(f"Envie no máximo {MAX_ANIMATION_ASSETS} imagens de objetos.")
        return False
    if form_data.get("voiceover_enabled") and not form_data.get("voice_api_key"):
        st.error("Informe a chave do provedor de voz antes de iniciar o projeto.")
        return False

    state.clear_progress()
    try:
        project = _persist_project_from_form(form_data)
    except ValueError as exc:
        st.error(str(exc))
        return False
    state.set(state.KEY_CURRENT_PROJECT_ID, project.id)
    state.set(
        state.KEY_GENERATION_METADATA,
        {
            "project_id": project.id,
            "status": project.status,
            "llm_provider": project.llm_provider,
            "llm_model": project.llm_model,
            "llm_api_key_source": project.llm_api_key_source,
            "voice_provider": project.voice_provider,
            "voice": project.voice,
            "voiceover_enabled": project.voiceover_enabled,
        },
    )
    state.append_progress("Projeto salvo no banco SQLite local.")
    state.append_progress(
        "Projeto pronto. Abra a página Gerar vídeos para preparar a base matemática e os planos."
    )
    return True


def _persist_project_from_form(form_data: Mapping[str, Any]) -> Any:
    """Persist form data without storing API keys."""
    from olympianim.services.project_service import (
        ProjectImageInput,
        ProjectInput,
        ProjectService,
    )

    problem_images = tuple(
        ProjectImageInput(filename=image.name, content=image.getvalue())
        for image in form_data.get("images", ())
    )
    solution_images = tuple(
        ProjectImageInput(filename=image.name, content=image.getvalue())
        for image in form_data.get("solution_images", ())
    )
    service = ProjectService()
    delivery_mode = service.repository.get_setting("output_delivery_mode", "separate")
    return service.create_project(
        ProjectInput(
            title=str(form_data.get("project_title", "")),
            problem_statement=str(form_data.get("statement", "")),
            problem_images=problem_images,
            problem_source=str(form_data.get("source", "")),
            problem_level=str(form_data.get("level", "")),
            math_area=str(form_data.get("area", "")),
            teacher_solution=str(form_data.get("teacher_solution", "")),
            solution_images=solution_images,
            teacher_instructions=str(form_data.get("teacher_instructions", "")),
            llm_provider=str(form_data.get("llm_provider", "")),
            llm_model=str(form_data.get("llm_model", "")),
            llm_api_key_source=str(form_data.get("llm_api_key_source", "")),
            voice_provider=str(form_data.get("voice_provider", "")),
            voice_model=str(form_data.get("voice_model", "")),
            voice=str(form_data.get("voice", "")),
            voice_language=str(form_data.get("language", "")),
            voice_speed=float(form_data.get("speed", 1.0)),
            voice_api_key_source=str(form_data.get("voice_api_key_source", "")),
            reuse_llm_api_key=bool(form_data.get("reuse_llm_api_key", False)),
            voiceover_enabled=bool(form_data.get("voiceover_enabled", False)),
            color_palette_id=str(form_data.get("color_palette_id", "")),
            color_palette_snapshot=str(form_data.get("color_palette_snapshot", "")),
            output_delivery_mode=delivery_mode,
            animation_assets=tuple(form_data.get("animation_assets", ())),
        )
    )
