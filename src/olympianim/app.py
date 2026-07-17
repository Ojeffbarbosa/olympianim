"""Olympianim - main Streamlit entry point.

Run with::

    streamlit run src/olympianim/app.py

This file is the navigation hub. It registers application pages with
Portuguese display titles and delegates rendering to the page scripts.
Business logic lives under ``src/olympianim`` so page scripts stay thin.
"""

from __future__ import annotations

import streamlit as st

from olympianim.ui import state


def main() -> None:
    """Assemble the navigation and run the selected page."""
    st.set_page_config(
        page_title="Olympianim",
        page_icon=":material/movie:",
        layout="wide",
    )
    state.init_defaults()

    pages = [
        st.Page(
            "pages/home.py",
            title="Início",
            icon=":material/home:",
            default=True,
        ),
        st.Page(
            "pages/projects.py",
            title="Projetos",
            icon=":material/folder:",
        ),
        st.Page(
            "pages/generate_videos.py",
            title="Gerar vídeos",
            icon=":material/movie:",
        ),
        st.Page(
            "pages/code_editor.py",
            title="Editor Manim",
            icon=":material/code:",
        ),
        st.Page(
            "pages/usage.py",
            title="Consumo",
            icon=":material/monitoring:",
        ),
        st.Page(
            "pages/prompts.py",
            title="Prompts",
            icon=":material/chat:",
        ),
        st.Page(
            "pages/settings.py",
            title="Configurações",
            icon=":material/settings:",
        ),
    ]

    navigation = st.navigation(pages)
    navigation.run()


if __name__ == "__main__":
    main()
