"""Progress and result presentation for the Streamlit home page."""

from __future__ import annotations

from typing import Any

import streamlit as st

from olympianim.ui import state


def render_progress_area() -> None:
    """Render the on-screen progress messages."""
    log: list[str] = state.get(state.KEY_PROGRESS_LOG, [])
    if not log:
        return
    with st.status("Progresso da geração", expanded=True) as status:
        for message in log:
            st.write(message)
        if any("final" in message.lower() or "conclu" in message.lower() for message in log):
            status.update(label="Geração concluída", state="complete")
        else:
            status.update(label="Gerando...", state="running")


def render_results_area() -> None:
    """Render available video players and generation metadata."""
    presentation_path: str = state.get(state.KEY_PRESENTATION_VIDEO, "")
    solution_path: str = state.get(state.KEY_SOLUTION_VIDEO, "")
    metadata: dict[str, Any] = state.get(state.KEY_GENERATION_METADATA, {})
    if not presentation_path and not solution_path and not metadata:
        return
    st.subheader("Resultado")
    if presentation_path:
        st.markdown("**Vídeo de apresentação**")
        st.video(presentation_path)
        st.caption(f"Caminho: `{presentation_path}`")
    if solution_path:
        st.markdown("**Vídeo de resolução**")
        st.video(solution_path)
        st.caption(f"Caminho: `{solution_path}`")
    if metadata:
        with st.expander("Metadados da geração"):
            st.json(metadata)
