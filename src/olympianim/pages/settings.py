"""Application defaults and configurable provider-model catalog."""

from __future__ import annotations

import streamlit as st

from olympianim.database.models import ColorPaletteRecord, ModelCatalogRecord
from olympianim.database.repository import ProjectRepository
from olympianim.services.code_assistant_preferences import (
    CodeAssistantPreferences,
    CodeAssistantPreferencesService,
)
from olympianim.services.color_palette import (
    ColorPaletteInput,
    ColorPaletteService,
)
from olympianim.services.model_catalog import (
    SUPPORTED_PROVIDERS,
    CatalogModelInput,
    ModelCatalogService,
)

_PALETTE_FIELDS = (
    ("background", "Fundo"),
    ("primary_text", "Texto principal"),
    ("secondary_text", "Texto secundário"),
    ("surface", "Superfície"),
    ("primary", "Primária"),
    ("secondary", "Secundária"),
    ("highlight", "Destaque"),
    ("stroke", "Contorno"),
)


def _catalog_input(
    record: ModelCatalogRecord | None,
    *,
    provider: str,
    modality: str,
    prefix: str,
) -> CatalogModelInput:
    model_id = st.text_input(
        "Identificador usado na API",
        value=record.model_id if record else "",
        key=f"{prefix}_model_id",
    )
    display_name = st.text_input(
        "Nome de exibição (opcional)",
        value=record.display_name if record else "",
        key=f"{prefix}_display_name",
    )
    enabled_column, default_column, order_column = st.columns(3, gap="medium")
    with enabled_column:
        enabled = st.checkbox(
            "Ativo",
            value=record.enabled if record else True,
            key=f"{prefix}_enabled",
        )
    with default_column:
        is_default = st.checkbox(
            "Modelo padrão",
            value=record.is_default if record else False,
            key=f"{prefix}_default",
        )
    with order_column:
        sort_order = st.number_input(
            "Ordem",
            min_value=0,
            step=1,
            value=record.sort_order if record else 0,
            key=f"{prefix}_order",
        )

    input_rate = cached_rate = output_rate = character_rate = audio_rate = 0.0
    if modality == "text":
        st.caption("Preços em USD por 1 milhão de tokens.")
        input_column, cache_column, output_column = st.columns(3, gap="medium")
        with input_column:
            input_rate = st.number_input(
                "Entrada",
                min_value=0.0,
                format="%.6f",
                value=record.input_token_rate if record else 0.0,
                key=f"{prefix}_input_rate",
            )
        with cache_column:
            cached_rate = st.number_input(
                "Entrada em cache",
                min_value=0.0,
                format="%.6f",
                value=record.cached_input_token_rate if record else 0.0,
                key=f"{prefix}_cache_rate",
            )
        with output_column:
            output_rate = st.number_input(
                "Saída",
                min_value=0.0,
                format="%.6f",
                value=record.output_token_rate if record else 0.0,
                key=f"{prefix}_output_rate",
            )
    else:
        st.caption("Preços em USD por 1 milhão da unidade indicada.")
        character_column, input_column, audio_column = st.columns(3, gap="medium")
        with character_column:
            character_rate = st.number_input(
                "Caracteres de entrada",
                min_value=0.0,
                format="%.6f",
                value=record.input_character_rate if record else 0.0,
                key=f"{prefix}_character_rate",
            )
        with input_column:
            input_rate = st.number_input(
                "Tokens de entrada",
                min_value=0.0,
                format="%.6f",
                value=record.input_token_rate if record else 0.0,
                key=f"{prefix}_input_rate",
            )
        with audio_column:
            audio_rate = st.number_input(
                "Tokens de áudio",
                min_value=0.0,
                format="%.6f",
                value=record.audio_output_token_rate if record else 0.0,
                key=f"{prefix}_audio_rate",
            )
    return CatalogModelInput(
        provider=provider,
        modality=modality,
        model_id=model_id,
        display_name=display_name,
        enabled=enabled,
        is_default=is_default,
        sort_order=int(sort_order),
        input_token_rate=float(input_rate),
        cached_input_token_rate=float(cached_rate),
        output_token_rate=float(output_rate),
        input_character_rate=float(character_rate),
        audio_output_token_rate=float(audio_rate),
    )


def _save_model(
    service: ModelCatalogService,
    data: CatalogModelInput,
    *,
    record_id: str | None = None,
) -> None:
    try:
        service.save(data, record_id=record_id)
    except ValueError as exc:
        st.error(str(exc))
        return
    st.success("Modelo salvo no catálogo.")
    st.rerun()


def _render_catalog(service: ModelCatalogService, modality: str) -> None:
    title = "Modelos de IA" if modality == "text" else "Modelos de voz"
    st.subheader(title)
    st.caption(
        "Modelos desativados permanecem disponíveis no histórico, mas não aparecem em novos projetos."
    )
    provider_options = SUPPORTED_PROVIDERS if modality == "text" else ("OpenAI", "Google")
    provider = st.segmented_control(
        "Provedor",
        provider_options,
        default=provider_options[0],
        key=f"catalog_provider_{modality}",
    )
    if provider is None:
        return
    models = service.list_models(provider=provider, modality=modality)
    st.dataframe(
        [
            {
                "Modelo": item.display_name or item.model_id,
                "Identificador API": item.model_id,
                "Estado": "Ativo" if item.enabled else "Desativado",
                "Padrão": "Sim" if item.is_default else "Não",
                "Preço": service.price_label(item),
            }
            for item in models
        ],
        width="stretch",
        hide_index=True,
    )

    if models:
        selected_id = st.selectbox(
            "Editar modelo",
            [item.id for item in models],
            format_func=lambda value: next(
                item.display_name or item.model_id for item in models if item.id == value
            ),
            key=f"catalog_selected_{modality}_{provider}",
        )
        selected = next(item for item in models if item.id == selected_id)
        with st.form(f"edit_catalog_{selected.id}", border=True):
            edited = _catalog_input(
                selected,
                provider=provider,
                modality=modality,
                prefix=f"edit_{selected.id}",
            )
            save = st.form_submit_button(
                "Salvar alterações",
                type="primary",
                icon=":material/save:",
            )
        if save:
            _save_model(service, edited, record_id=selected.id)

        action_columns = st.columns(2, gap="medium")
        if selected.enabled and action_columns[0].button(
            "Desativar",
            key=f"deactivate_{selected.id}",
            icon=":material/block:",
            width="stretch",
        ):
            try:
                service.deactivate(selected.id)
            except ValueError as exc:
                st.error(str(exc))
                return
            st.rerun()
        if service.can_restore_builtin(selected) and action_columns[1].button(
            "Restaurar padrão do app",
            key=f"restore_{selected.id}",
            icon=":material/restore:",
            width="stretch",
        ):
            service.restore_builtin(selected.id)
            st.rerun()

    with st.expander("Adicionar modelo", expanded=not models):
        with st.form(f"add_catalog_{modality}_{provider}", border=False):
            new_model = _catalog_input(
                None,
                provider=provider,
                modality=modality,
                prefix=f"add_{modality}_{provider}",
            )
            add = st.form_submit_button(
                "Adicionar modelo",
                type="primary",
                icon=":material/add:",
            )
        if add:
            _save_model(service, new_model)


def _palette_input(
    record: ColorPaletteRecord | None,
    *,
    prefix: str,
) -> ColorPaletteInput:
    """Render semantic palette fields inside the caller's form."""
    defaults = ColorPaletteInput(name="")
    name = st.text_input(
        "Nome",
        value=record.name if record else "",
        key=f"{prefix}_name",
    )
    description = st.text_input(
        "Descrição (opcional)",
        value=record.description if record else "",
        key=f"{prefix}_description",
    )
    values: dict[str, str] = {}
    for row in range(0, len(_PALETTE_FIELDS), 4):
        columns = st.columns(4, gap="medium")
        for column, (field, label) in zip(columns, _PALETTE_FIELDS[row : row + 4], strict=True):
            with column:
                values[field] = st.color_picker(
                    label,
                    value=getattr(record or defaults, field),
                    key=f"{prefix}_{field}",
                )
    enabled_column, order_column = st.columns(2, gap="medium")
    with enabled_column:
        enabled = st.checkbox(
            "Ativa",
            value=record.enabled if record else True,
            key=f"{prefix}_enabled",
        )
    with order_column:
        sort_order = st.number_input(
            "Ordem",
            min_value=0,
            step=1,
            value=record.sort_order if record else 10,
            key=f"{prefix}_order",
        )
    return ColorPaletteInput(
        name=name,
        description=description,
        enabled=enabled,
        sort_order=int(sort_order),
        **values,
    )


def _palette_preview(record: ColorPaletteRecord) -> None:
    swatches = "".join(
        f'<span title="{label}: {getattr(record, field)}" '
        f'style="display:inline-block;width:36px;height:36px;'
        f"background:{getattr(record, field)};border:1px solid #888;"
        f'margin-right:8px"></span>'
        for field, label in _PALETTE_FIELDS
    )
    st.html(f'<div aria-label="Prévia da paleta">{swatches}</div>')


def _render_palette_catalog(service: ColorPaletteService) -> None:
    st.subheader("Paletas visuais")
    st.caption(
        "Paletas selecionadas orientam os builders e o corretor. "
        "O modo automático não envia instruções de cor à IA."
    )
    palettes = service.list_palettes()
    selected_id = st.selectbox(
        "Visualizar ou editar paleta",
        options=[item.id for item in palettes],
        format_func=lambda value: next(item.name for item in palettes if item.id == value),
        key="selected_color_palette",
    )
    selected = next(item for item in palettes if item.id == selected_id)
    _palette_preview(selected)
    st.caption(selected.description)

    if selected.is_builtin:
        st.info("Esta paleta é fornecida pelo Olympianim. Duplique-a para personalizar.")
        if st.button(
            "Duplicar para editar",
            type="primary",
            icon=":material/content_copy:",
            width="stretch",
        ):
            service.duplicate(selected.id)
            st.rerun()
    else:
        with st.form(f"edit_palette_{selected.id}", border=True):
            edited = _palette_input(selected, prefix=f"edit_palette_{selected.id}")
            save = st.form_submit_button(
                "Salvar alterações",
                type="primary",
                icon=":material/save:",
            )
        if save:
            try:
                service.save(edited, record_id=selected.id)
            except ValueError as exc:
                st.error(str(exc))
            else:
                st.success("Paleta salva.")
                st.rerun()
        if selected.enabled and st.button(
            "Desativar paleta",
            key=f"deactivate_palette_{selected.id}",
            icon=":material/block:",
            width="stretch",
        ):
            service.deactivate(selected.id)
            st.rerun()

    with st.expander("Adicionar paleta"):
        with st.form("add_color_palette", border=False):
            new_palette = _palette_input(None, prefix="add_palette")
            add = st.form_submit_button(
                "Adicionar paleta",
                type="primary",
                icon=":material/add:",
            )
        if add:
            try:
                service.save(new_palette)
            except ValueError as exc:
                st.error(str(exc))
            else:
                st.success("Paleta adicionada.")
                st.rerun()


st.title("Configurações")
repository = ProjectRepository()
catalog_service = ModelCatalogService(repository)
palette_service = ColorPaletteService(repository)
general_tab, llm_tab, voice_tab, palette_tab = st.tabs(
    ["Geral", "Modelos de IA", "Modelos de voz", "Paletas"],
    key="settings_tabs",
    on_change="rerun",
)

with general_tab:
    current = repository.get_setting("output_delivery_mode", "separate")
    selected = str(
        st.radio(
            "Entrega padrão para novos projetos",
            options=("separate", "combined"),
            format_func=lambda value: "Vídeos separados" if value == "separate" else "Vídeo único",
            index=0 if current == "separate" else 1,
        )
    )
    quality_options = (
        "low_quality",
        "medium_quality",
        "high_quality",
        "production_quality",
    )
    quality_labels = {
        "low_quality": "Baixa (480p, desenvolvimento)",
        "medium_quality": "Média (720p)",
        "high_quality": "Alta (1080p)",
        "production_quality": "Produção (2160p)",
    }
    current_quality = repository.get_setting("render_quality", "low_quality")
    quality = str(
        st.selectbox(
            "Qualidade de renderização",
            options=quality_options,
            format_func=lambda value: quality_labels[value],
            index=(
                quality_options.index(current_quality) if current_quality in quality_options else 0
            ),
        )
    )
    if st.button(
        "Salvar configuração",
        type="primary",
        icon=":material/save:",
    ):
        repository.set_setting("output_delivery_mode", selected)
        repository.set_setting("render_quality", quality)
        st.success("Configurações salvas.")

    st.divider()
    st.subheader("Assistente Manim")
    st.caption("Modelo padrão compartilhado pelos modos Conversa e Editor.")
    assistant_service = CodeAssistantPreferencesService(repository)
    assistant_default = assistant_service.resolve()
    assistant_providers = catalog_service.providers("text")
    if not assistant_providers:
        st.error("Ative ao menos um modelo de IA no catálogo.")
    else:
        provider_key = "settings_code_assistant_provider"
        model_key = "settings_code_assistant_model"
        if st.session_state.get(provider_key) not in assistant_providers:
            st.session_state[provider_key] = assistant_default.provider
        assistant_provider = st.selectbox(
            "Provedor padrão do assistente",
            assistant_providers,
            key=provider_key,
        )
        assistant_models = catalog_service.model_ids(assistant_provider, "text")
        if st.session_state.get(model_key) not in assistant_models:
            preferred_model = (
                assistant_default.model
                if assistant_provider == assistant_default.provider
                else catalog_service.default_model_id(assistant_provider, "text")
            )
            st.session_state[model_key] = preferred_model
        assistant_model = st.selectbox(
            "Modelo padrão do assistente",
            assistant_models,
            format_func=lambda value: catalog_service.label(
                assistant_provider,
                "text",
                value,
            ),
            key=model_key,
        )
        if st.button(
            "Salvar padrão do assistente",
            type="primary",
            icon=":material/save:",
        ):
            try:
                assistant_service.save(
                    CodeAssistantPreferences(
                        provider=assistant_provider,
                        model=assistant_model,
                    )
                )
            except ValueError as exc:
                st.error(str(exc))
            else:
                st.success("Modelo padrão do assistente salvo.")

with llm_tab:
    _render_catalog(catalog_service, "text")

with voice_tab:
    _render_catalog(catalog_service, "speech")

with palette_tab:
    _render_palette_catalog(palette_service)
