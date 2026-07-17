"""Home page of the Olympianim teacher interface.

This script is loaded via ``st.Page`` from ``app.py``. It renders the
problem input form, the AI, voice and visual configuration selectors,
the generation button, the progress area and the results area. Page scripts stay thin;
rendering helpers live in ``olympianim.ui.sections``.
"""

from __future__ import annotations

import streamlit as st

from olympianim.ui import sections

sections.render_header()

with st.expander("Configuração da geração", expanded=False):
    col_llm, col_voice = st.columns(2)
    with col_llm:
        llm = sections.render_llm_config_section()
    with col_voice:
        voice = sections.render_voice_config_section()
    palette = sections.render_color_palette_section()
    preferences = sections.capture_generation_preferences(llm, voice, palette)
    sections.render_save_generation_preferences_button(preferences)

animation_assets = sections.render_animation_assets_section()

with st.form("project_creation_form", clear_on_submit=False):
    st.subheader("Projeto")
    problem = sections.render_problem_section()
    teacher = sections.render_teacher_extras_section()
    form_data: dict[str, object] = {
        **problem,
        **teacher,
        **llm,
        **voice,
        **palette,
        **animation_assets,
    }
    project_created = sections.render_generate_button(form_data)

if project_created:
    st.switch_page("pages/generate_videos.py")
else:
    sections.render_progress_area()

sections.render_results_area()
