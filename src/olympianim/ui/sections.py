"""Render functions for each section of the Streamlit home page.

Every function here renders a single visual block and returns the
captured values. The page file (``src/olympianim/app.py``) merely calls
them in order, keeping business logic out of the page.

All user-facing text is in Portuguese; identifiers and
docstrings remain in English.
"""

from __future__ import annotations

import base64
from collections.abc import Mapping
from typing import Any

import streamlit as st

from olympianim.config import APP_NAME, LOGO_LIGHT_PATH
from olympianim.services.generation_preferences import (
    GenerationPreferences,
    GenerationPreferencesService,
)
from olympianim.services.image_asset_service import (
    MAX_ANIMATION_ASSETS,
    AnimationAssetInput,
    ImageAssetService,
)
from olympianim.ui import options as opt
from olympianim.ui import state
from olympianim.ui.generation_results import (
    render_progress_area as render_progress_area,
)
from olympianim.ui.generation_results import render_results_area as render_results_area
from olympianim.ui.project_form import render_generate_button as render_generate_button


def render_header() -> None:
    """Render the application title and a short description."""
    _render_theme_aware_logo()
    st.divider()


def _render_theme_aware_logo() -> None:
    """Render a logo that follows the browser's active Streamlit color scheme."""
    logo_data = base64.b64encode(LOGO_LIGHT_PATH.read_bytes()).decode("ascii")
    st.html(f"""
        <style>
        .olympianim-brand {{
            display: flex;
            align-items: center;
            gap: 12px;
            min-width: 0;
        }}
        .olympianim-logo {{
            flex: 0 0 84px;
            width: 84px;
            height: 84px;
            background-color: #000000;
            background-color: light-dark(#000000, #ffffff);
            -webkit-mask: url("data:image/png;base64,{logo_data}") center / contain no-repeat;
            mask: url("data:image/png;base64,{logo_data}") center / contain no-repeat;
        }}
        .olympianim-brand-copy {{
            min-width: 0;
        }}
        .olympianim-brand-title {{
            margin: 0;
            font-size: 2.5rem;
            line-height: 1.2;
        }}
        .olympianim-brand-caption {{
            margin: 0.2rem 0 0;
            font-size: 0.9rem;
            line-height: 1.45;
        }}
        @media (max-width: 640px) {{
            .olympianim-brand {{
                align-items: flex-start;
            }}
            .olympianim-logo {{
                flex-basis: 64px;
                width: 64px;
                height: 64px;
            }}
            .olympianim-brand-title {{
                font-size: 2rem;
            }}
        }}
        </style>
        <div class="olympianim-brand">
            <div class="olympianim-logo" role="img" aria-label="Olympianim"></div>
            <div class="olympianim-brand-copy">
                <h1 class="olympianim-brand-title">{APP_NAME}</h1>
                <p class="olympianim-brand-caption">
                    Ferramenta local de autoria docente para produzir vídeos educacionais de
                    problemas olímpicos com IA, Manim e narração opcional.
                </p>
            </div>
        </div>
        """)


def render_problem_section() -> Mapping[str, Any]:
    """Render the main problem input block.

    Returns a mapping with the captured fields.
    """
    project_title = st.text_input(
        "Nome do projeto",
        max_chars=120,
        placeholder="Ex.: Tabuleiro com somas iguais - OBMEP 2024",
        key=state.KEY_PROJECT_TITLE,
    )

    st.subheader("Enunciado do problema")

    statement = st.text_area(
        "Enunciado do problema",
        height=180,
        placeholder="Cole aqui o enunciado completo do problema olímpico...",
        key=state.KEY_PROBLEM_STATEMENT,
    )

    images = st.file_uploader(
        "Imagens da questão (opcional)",
        type=["png", "jpg", "jpeg", "webp"],
        accept_multiple_files=True,
        key=state.KEY_PROBLEM_IMAGE,
        help="Envie na ordem em que as páginas ou partes devem ser lidas.",
    )

    col_src, col_lvl, col_area = st.columns(3)
    with col_src:
        source = st.text_input(
            "Fonte (opcional)",
            placeholder="Ex.: OBMEP 2023",
            key=state.KEY_PROBLEM_SOURCE,
        )
    with col_lvl:
        level = st.text_input(
            "Ano / Nível (opcional)",
            placeholder="Ex.: Nível 2",
            key=state.KEY_PROBLEM_LEVEL,
        )
    with col_area:
        area = st.selectbox(
            "Área matemática",
            options=opt.MATH_AREAS,
            key=state.KEY_MATH_AREA,
        )

    return {
        "project_title": project_title,
        "statement": statement,
        "images": tuple(images),
        "source": source,
        "level": level,
        "area": area,
    }


def render_animation_assets_section() -> Mapping[str, Any]:
    """Render object uploads outside the project form so metadata updates immediately."""
    with st.expander("Objetos visuais para animação (opcional)", expanded=False):
        uploads = st.file_uploader(
            "Imagens de objetos",
            type=["png", "jpg", "jpeg", "webp"],
            accept_multiple_files=True,
            key="animation_asset_uploads",
        )
        animation_assets: list[AnimationAssetInput] = []
        for index, upload in enumerate(uploads[:MAX_ANIMATION_ASSETS], start=1):
            st.markdown(f"**{index}. {upload.name}**")
            preview_column, fields_column = st.columns([1, 2])
            with preview_column:
                st.image(upload, width="stretch")
            with fields_column:
                description = st.text_area(
                    "Descrição obrigatória",
                    key=f"animation_asset_description_{index}_{upload.name}",
                    height=90,
                    placeholder="Ex.: motocicleta vermelha vista de perfil, apontada para a direita.",
                )
                remove_background = st.checkbox(
                    "Remover fundo uniforme",
                    key=f"animation_asset_remove_background_{index}_{upload.name}",
                )
            asset = AnimationAssetInput(
                filename=upload.name,
                content=upload.getvalue(),
                description=description,
                remove_background=remove_background,
            )
            animation_assets.append(asset)
            if remove_background:
                preview_asset = AnimationAssetInput(
                    filename=asset.filename,
                    content=asset.content,
                    description=(
                        asset.description
                        if len(asset.description.strip()) >= 10
                        else "Descrição temporária para gerar a prévia."
                    ),
                    remove_background=True,
                )
                try:
                    prepared = ImageAssetService().prepare_many((preview_asset,))[0]
                except ValueError as exc:
                    st.error(str(exc))
                else:
                    st.image(prepared.content, caption="Prévia sem fundo", width=240)
        if len(uploads) > MAX_ANIMATION_ASSETS:
            st.error(f"Envie no máximo {MAX_ANIMATION_ASSETS} imagens de objetos.")
    return {
        "animation_assets": tuple(animation_assets),
        "animation_asset_upload_count": len(uploads),
    }


def render_teacher_extras_section() -> Mapping[str, Any]:
    """Render optional teacher solution and instructions."""
    with st.expander("Resolução e instruções do professor (opcional)", expanded=False):
        st.markdown(
            "Você pode enviar sua própria resolução e orientações livres "
            "para condicionar a geração. Esses campos são opcionais."
        )

        teacher_solution = st.text_area(
            "Resolução do professor (opcional)",
            height=140,
            placeholder="Cole aqui sua resolução, parcial ou apenas a ideia principal...",
            key=state.KEY_TEACHER_SOLUTION,
        )

        solution_images = st.file_uploader(
            "Imagens da resolução (opcional)",
            type=["png", "jpg", "jpeg", "webp"],
            accept_multiple_files=True,
            key=state.KEY_SOLUTION_IMAGES,
            help="Envie fotos do caderno ou páginas da solução oficial na ordem de leitura.",
        )

        teacher_instructions = st.text_area(
            "Instruções adicionais para a IA (opcional)",
            height=100,
            placeholder=(
                "Ex.: use abordagem geométrica; evite álgebra pesada; "
                "narração em tom de aula para ensino fundamental."
            ),
            key=state.KEY_TEACHER_INSTRUCTIONS,
        )

    return {
        "teacher_solution": teacher_solution,
        "solution_images": tuple(solution_images),
        "teacher_instructions": teacher_instructions,
    }


def render_llm_config_section() -> Mapping[str, Any]:
    """Render the language-model configuration block."""
    from olympianim.ui import credentials as cred

    st.subheader("Configuração da IA")

    providers = opt.active_llm_providers()
    if not providers:
        st.error("Ative ao menos um modelo de IA em Configurações.")
        return {
            "llm_provider": "",
            "llm_model": "",
            "llm_api_key": "",
            "llm_api_key_source": "",
            "connection_ok": False,
            "connection_tested": False,
        }
    if state.get(state.KEY_LLM_PROVIDER, providers[0]) not in providers:
        state.set(state.KEY_LLM_PROVIDER, providers[0])
    provider = st.selectbox(
        "Provedor de IA",
        options=providers,
        key=state.KEY_LLM_PROVIDER,
    )

    available_models = opt.models_for(provider)
    current_model = state.get(state.KEY_LLM_MODEL, available_models[0] if available_models else "")
    if current_model not in available_models and available_models:
        state.set(state.KEY_LLM_MODEL, available_models[0])
    model = st.selectbox(
        "Modelo",
        options=available_models,
        format_func=lambda value: opt.model_label(provider, value),
        key=state.KEY_LLM_MODEL,
    )

    # Resolve any key already known (.env first), then let the teacher
    # type/override it. The typed value is kept only in memory.
    store = cred.get_credential_store()
    pre_resolved = store.resolve_llm(provider)
    _ = pre_resolved  # consulted to surface source badge below
    api_key = st.text_input(
        "Chave de API",
        type="password",
        placeholder=cred.key_hint_text(provider),
        key=state.KEY_LLM_API_KEY,
        help=cred.key_help_text(),
    )

    resolved = cred.resolve_llm_key(provider, api_key)

    st.caption(f"Origem da chave: **{cred.key_source_badge(resolved)}**")
    st.warning(cred.session_warning_text_from_service())

    conn = cred.render_connection_test(provider, resolved)

    return {
        "llm_provider": provider,
        "llm_model": model,
        "llm_api_key": resolved.value,
        "llm_api_key_source": resolved.source,
        "connection_ok": conn["ok"],
        "connection_tested": conn["tested"],
    }


def render_voice_config_section() -> Mapping[str, Any]:
    """Render the optional voiceover configuration block."""
    st.subheader("Configuração de voz")

    enabled = st.checkbox(
        "Adicionar narração ao vídeo",
        key=state.KEY_VOICEOVER_ENABLED,
    )
    if not enabled:
        return {
            "voiceover_enabled": False,
            "voice_provider": "",
            "voice_model": "",
            "voice": "",
            "language": "",
            "speed": 1.0,
            "voice_api_key": "",
            "voice_api_key_source": "",
            "reuse_llm_api_key": False,
        }

    from olympianim.ui import credentials as cred

    providers = opt.active_voice_providers()
    if not providers:
        st.error("Ative ao menos um modelo de voz em Configurações.")
        return {
            "voiceover_enabled": True,
            "voice_provider": "",
            "voice_model": "",
            "voice": "",
            "language": "",
            "speed": 1.0,
            "voice_api_key": "",
            "voice_api_key_source": "",
            "reuse_llm_api_key": False,
        }
    if state.get(state.KEY_VOICE_PROVIDER, providers[0]) not in providers:
        state.set(state.KEY_VOICE_PROVIDER, providers[0])
    provider = st.selectbox(
        "Provedor de voz",
        options=providers,
        key=state.KEY_VOICE_PROVIDER,
    )

    available_models = opt.voice_models_for(provider)
    current_model = state.get(
        state.KEY_VOICE_MODEL, available_models[0] if available_models else ""
    )
    if current_model not in available_models and available_models:
        state.set(state.KEY_VOICE_MODEL, available_models[0])
    model = st.selectbox(
        "Modelo de voz",
        options=available_models,
        format_func=lambda value: opt.model_label(provider, value, modality="speech"),
        key=state.KEY_VOICE_MODEL,
    )

    available_voices = opt.voices_for(provider)
    current_voice = state.get(state.KEY_VOICE, available_voices[0] if available_voices else "")
    if current_voice not in available_voices and available_voices:
        state.set(state.KEY_VOICE, available_voices[0])
    voice = st.selectbox(
        "Voz",
        options=available_voices,
        key=state.KEY_VOICE,
    )

    language = st.selectbox(
        "Idioma",
        options=opt.VOICE_LANGUAGES,
        key=state.KEY_VOICE_LANGUAGE,
    )

    speed = st.slider(
        "Velocidade da narração",
        min_value=0.5,
        max_value=2.0,
        step=0.1,
        help="1.0 é a velocidade normal. Nem todos os provedores suportam ajuste.",
        key=state.KEY_VOICE_SPEED,
    )

    llm_provider = str(state.get(state.KEY_LLM_PROVIDER, ""))
    can_reuse = provider == llm_provider
    reuse_llm_key = False
    if can_reuse:
        reuse_llm_key = st.checkbox(
            "Usar a mesma chave da IA",
            key=state.KEY_REUSE_LLM_API_KEY,
        )

    if reuse_llm_key:
        voice_api_key = str(state.get(state.KEY_API_KEY_RESOLVED, ""))
        voice_api_key_source = "reused_llm" if voice_api_key else ""
        st.caption("A narração usará a chave já resolvida para a IA.")
    else:
        typed_voice_key = st.text_input(
            "Chave de API da voz",
            type="password",
            placeholder=cred.voice_key_hint_text(provider),
            key=state.KEY_VOICE_API_KEY,
            help=cred.key_help_text(),
        )
        resolved_voice = cred.resolve_voice_key(provider, typed_voice_key)
        voice_api_key = resolved_voice.value
        voice_api_key_source = resolved_voice.source
        st.caption(f"Origem da chave de voz: **{cred.key_source_badge(resolved_voice)}**")

    return {
        "voiceover_enabled": True,
        "voice_provider": provider,
        "voice_model": model,
        "voice": voice,
        "language": language,
        "speed": speed,
        "voice_api_key": voice_api_key,
        "voice_api_key_source": voice_api_key_source,
        "reuse_llm_api_key": reuse_llm_key,
    }


def render_color_palette_section() -> Mapping[str, Any]:
    """Render the optional semantic palette selector."""
    from olympianim.services.color_palette import ColorPaletteService

    service = ColorPaletteService()
    palettes = service.list_palettes(enabled_only=True)
    palette_ids = ("", *(palette.id for palette in palettes))
    current = str(state.get(state.KEY_COLOR_PALETTE_ID, ""))
    if current not in palette_ids:
        state.set(state.KEY_COLOR_PALETTE_ID, "")

    st.subheader("Paleta visual")
    palette_id = st.selectbox(
        "Cores da animação",
        options=palette_ids,
        format_func=lambda value: (
            "Automática (IA decide)"
            if not value
            else next(item.name for item in palettes if item.id == value)
        ),
        key=state.KEY_COLOR_PALETTE_ID,
        help="No modo automático, nenhuma instrução de cor é enviada à IA.",
    )
    selected = next((item for item in palettes if item.id == palette_id), None)
    if selected is not None:
        colors = (
            selected.background,
            selected.primary_text,
            selected.secondary_text,
            selected.surface,
            selected.primary,
            selected.secondary,
            selected.highlight,
            selected.stroke,
        )
        swatches = "".join(
            f'<span style="display:inline-block;width:28px;height:28px;'
            f'background:{color};border:1px solid #888;margin-right:6px"></span>'
            for color in colors
        )
        st.html(f'<div aria-label="Amostras da paleta">{swatches}</div>')
        st.caption(selected.description)
    return {
        "color_palette_id": palette_id,
        "color_palette_snapshot": service.snapshot(palette_id),
    }


def capture_generation_preferences(
    llm: Mapping[str, Any],
    voice: Mapping[str, Any],
    palette: Mapping[str, Any],
) -> GenerationPreferences:
    """Keep a complete non-sensitive draft independent from widget lifecycle."""
    previous = state.get(
        state.KEY_GENERATION_PREFERENCES_DRAFT,
        GenerationPreferences(),
    )
    voice_enabled = bool(voice.get("voiceover_enabled", False))
    preferences = GenerationPreferences(
        llm_provider=str(llm.get("llm_provider", previous.llm_provider)),
        llm_model=str(llm.get("llm_model", previous.llm_model)),
        voiceover_enabled=voice_enabled,
        voice_provider=(
            str(voice.get("voice_provider", previous.voice_provider))
            if voice_enabled
            else previous.voice_provider
        ),
        voice_model=(
            str(voice.get("voice_model", previous.voice_model))
            if voice_enabled
            else previous.voice_model
        ),
        voice=(str(voice.get("voice", previous.voice)) if voice_enabled else previous.voice),
        voice_language=(
            str(voice.get("language", previous.voice_language))
            if voice_enabled
            else previous.voice_language
        ),
        voice_speed=(
            float(voice.get("speed", previous.voice_speed))
            if voice_enabled
            else previous.voice_speed
        ),
        reuse_llm_api_key=(
            bool(voice.get("reuse_llm_api_key", previous.reuse_llm_api_key))
            if voice_enabled
            else previous.reuse_llm_api_key
        ),
        color_palette_id=str(palette.get("color_palette_id", previous.color_palette_id)),
    )
    state.set(state.KEY_GENERATION_PREFERENCES_DRAFT, preferences)
    return preferences


def render_save_generation_preferences_button(
    preferences: GenerationPreferences,
) -> None:
    """Persist current non-sensitive selections as defaults for new projects."""
    if st.button(
        "Salvar configurações para próximos projetos",
        type="primary",
        icon=":material/save:",
        width="stretch",
    ):
        GenerationPreferencesService().save(preferences)
        st.success("Configurações salvas para os próximos projetos.")
