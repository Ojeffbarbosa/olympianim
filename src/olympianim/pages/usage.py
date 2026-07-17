"""Project-level LLM consumption dashboard."""

from __future__ import annotations

from datetime import date

import streamlit as st

from olympianim.services.project_service import ProjectService
from olympianim.services.usage_service import UsageFilters, UsageService, UsageTotals

_AGENT_LABELS = {
    "planner": "Planejador",
    "builder": "Builder",
    "solver": "Solucionador",
    "debugger": "Corretor",
    "voice": "Narração",
}
_STAGE_LABELS = {
    "presentation": "Apresentação",
    "solution": "Resolução",
}


def _format_tokens(value: int) -> str:
    return f"{value:,}".replace(",", ".")


def _format_cost(value: float) -> str:
    if value == 0:
        return "$0"
    if value < 0.01:
        return f"${value:.6f}"
    return f"${value:,.2f}"


def _chart_rows(groups: dict[str, UsageTotals], label: str) -> list[dict[str, object]]:
    return [
        {
            label: key,
            "Entrada": totals.input_tokens,
            "Saída": totals.output_tokens,
            "Raciocínio": totals.reasoning_tokens,
        }
        for key, totals in groups.items()
    ]


def _cost_rows(groups: dict[str, UsageTotals], label: str) -> list[dict[str, object]]:
    return [
        {label: key, "Custo estimado (USD)": totals.estimated_cost_usd}
        for key, totals in groups.items()
    ]


st.title("Consumo de IA")
st.caption(
    "Chamadas e unidades de consumo informadas pelos provedores. Os custos são "
    "estimativas aproximadas calculadas com os preços cadastrados no aplicativo e "
    "podem divergir da cobrança efetiva; consulte o painel do provedor."
)

projects = ProjectService().list_projects()
if not projects:
    st.info("Crie um projeto para acompanhar o consumo de IA.")
    st.stop()

project_titles = {project.id: project.title for project in projects}
scope_options = ["__all__", *project_titles]
selected_scope = st.selectbox(
    "Escopo",
    scope_options,
    index=0,
    format_func=lambda value: "Todos os projetos" if value == "__all__" else project_titles[value],
)
project_id = None if selected_scope == "__all__" else selected_scope
usage_service = UsageService()
all_records = (
    usage_service.list_all_usage()
    if project_id is None
    else usage_service.list_project_usage(project_id)
)
if not all_records:
    st.info("O escopo selecionado ainda não possui dados de consumo registrados.")
    st.stop()

available_providers = sorted({record.provider for record in all_records})
available_models = sorted({record.model for record in all_records})
available_agents = sorted({record.agent_type for record in all_records})
available_stages = sorted({record.stage for record in all_records})
available_modalities = sorted({record.modality for record in all_records})
minimum_date = date.fromisoformat(all_records[0].created_at[:10])
maximum_date = date.fromisoformat(all_records[-1].created_at[:10])

with st.expander("Filtros", expanded=False):
    selected_projects: list[str] = []
    if project_id is None:
        selected_projects = st.multiselect(
            "Projetos",
            list(project_titles),
            format_func=lambda value: project_titles[value],
        )
    period_start, period_end = st.columns(2)
    with period_start:
        start_date = st.date_input("Data inicial", value=minimum_date)
    with period_end:
        end_date = st.date_input("Data final", value=maximum_date)
    provider_filter, model_filter = st.columns(2)
    with provider_filter:
        providers = st.multiselect("Provedores", available_providers)
    with model_filter:
        models = st.multiselect("Modelos", available_models)
    agent_filter, stage_filter = st.columns(2)
    with agent_filter:
        agents = st.multiselect(
            "Agentes",
            available_agents,
            format_func=lambda value: _AGENT_LABELS.get(value, value),
        )
    with stage_filter:
        stages = st.multiselect(
            "Etapas",
            available_stages,
            format_func=lambda value: _STAGE_LABELS.get(value, value),
        )
    modalities = st.multiselect(
        "Modalidades",
        available_modalities,
        format_func=lambda value: "Voz" if value == "speech" else "Texto/multimodal",
    )

records = usage_service.filtered_project_usage(
    project_id,
    UsageFilters(
        project_ids=frozenset(selected_projects),
        providers=frozenset(providers),
        models=frozenset(models),
        agents=frozenset(agents),
        stages=frozenset(stages),
        modalities=frozenset(modalities),
        start_date=start_date,
        end_date=end_date,
    ),
)
totals = usage_service.totals(records)
text_records = [record for record in records if record.modality == "text"]
text_totals = usage_service.totals(text_records)

if not records:
    st.warning("Nenhuma chamada corresponde aos filtros selecionados.")
    st.stop()

provider_groups = usage_service.grouped_totals(records, "provider")
text_provider_groups = usage_service.grouped_totals(text_records, "provider")
voice_records = [record for record in records if record.modality == "speech"]
summary_columns = st.columns(4, gap="medium", border=True)
summary_columns[0].metric("Chamadas", len(records))
summary_columns[1].metric("Concluídas", totals.completed_calls)
summary_columns[2].metric("Com falha", totals.failed_calls)
summary_columns[3].metric(
    "Custo aproximado (USD)",
    _format_cost(totals.estimated_cost_usd),
)

if totals.missing_metadata_calls or totals.unknown_cost_calls:
    with st.expander("Qualidade dos dados", expanded=False):
        if totals.missing_metadata_calls:
            st.info(
                f"{totals.missing_metadata_calls} chamada(s) não tiveram contagem "
                "informada pelo provedor. Tokens ausentes não são estimados."
            )
        if totals.unknown_cost_calls:
            st.info(
                f"{totals.unknown_cost_calls} chamada(s) ficaram sem custo estimado por "
                "preço não cadastrado ou ausência de métrica faturável após falha."
            )

overview_tab, text_tab, voice_tab, provider_tab, calls_tab = st.tabs(
    ["Visão geral", "Texto", "Voz", "Provedores", "Chamadas"],
    key="usage_dashboard_tabs",
    on_change="rerun",
)

with overview_tab:
    st.subheader("Distribuição do consumo")
    overview_left, overview_right = st.columns(2, gap="large")
    with overview_left:
        st.markdown("**Chamadas por provedor**")
        st.table(
            [
                {
                    "Provedor": provider,
                    "Chamadas": values.completed_calls + values.failed_calls,
                    "Falhas": values.failed_calls,
                    "Custo estimado (USD)": (
                        values.estimated_cost_usd if values.estimated_cost_usd else None
                    ),
                }
                for provider, values in provider_groups.items()
            ],
        )
    with overview_right:
        st.markdown("**Custo por provedor**")
        if totals.estimated_cost_usd:
            st.bar_chart(
                _cost_rows(provider_groups, "Provedor"),
                x="Provedor",
                y="Custo estimado (USD)",
                color="#2563EB",
                height=280,
            )
        else:
            st.info("Ainda não há chamadas com preço cadastrado neste recorte.")

    daily_groups = usage_service.grouped_totals(records, "day")
    if len(daily_groups) > 1 and totals.estimated_cost_usd:
        st.markdown("**Evolução diária do custo**")
        st.line_chart(
            _cost_rows(daily_groups, "Data"),
            x="Data",
            y="Custo estimado (USD)",
            color="#2563EB",
            height=280,
        )

    if project_id is None:
        project_groups = usage_service.grouped_totals(records, "project_id")
        if len(project_groups) > 1:
            st.markdown("**Comparação entre projetos**")
            st.bar_chart(
                _chart_rows(
                    {
                        project_titles.get(group_id, group_id): values
                        for group_id, values in project_groups.items()
                    },
                    "Projeto",
                ),
                x="Projeto",
                y=["Entrada", "Saída", "Raciocínio"],
                horizontal=True,
                stack=True,
                color=["#2563EB", "#16A34A", "#D97706"],
                height=320,
            )

with text_tab:
    if not text_records:
        st.info("Não há chamadas de texto ou multimodais neste recorte.")
    else:
        token_columns = st.columns(4, gap="medium", border=True)
        token_columns[0].metric("Entrada", _format_tokens(text_totals.input_tokens))
        token_columns[1].metric("Saída", _format_tokens(text_totals.output_tokens))
        token_columns[2].metric("Cache", _format_tokens(text_totals.cache_read_tokens))
        token_columns[3].metric("Raciocínio", _format_tokens(text_totals.reasoning_tokens))
        st.space("small")
        text_left, text_right = st.columns(2, gap="large")
        with text_left:
            st.markdown("**Por provedor**")
            st.bar_chart(
                _chart_rows(text_provider_groups, "Provedor"),
                x="Provedor",
                y=["Entrada", "Saída", "Raciocínio"],
                stack=True,
                color=["#2563EB", "#16A34A", "#D97706"],
                height=320,
            )
        with text_right:
            text_day_groups = usage_service.grouped_totals(text_records, "day")
            if len(text_day_groups) > 1:
                st.markdown("**Evolução diária**")
                st.line_chart(
                    _chart_rows(text_day_groups, "Data"),
                    x="Data",
                    y=["Entrada", "Saída", "Raciocínio"],
                    color=["#2563EB", "#16A34A", "#D97706"],
                    height=320,
                )
            else:
                st.markdown("**Por agente**")
                st.bar_chart(
                    _chart_rows(
                        {
                            _AGENT_LABELS.get(agent, agent): values
                            for agent, values in usage_service.grouped_totals(
                                text_records, "agent_type"
                            ).items()
                        },
                        "Agente",
                    ),
                    x="Agente",
                    y=["Entrada", "Saída", "Raciocínio"],
                    horizontal=True,
                    stack=True,
                    color=["#2563EB", "#16A34A", "#D97706"],
                    height=320,
                )

with voice_tab:
    if not voice_records:
        st.info("Ainda não há chamadas de narração neste recorte.")
    else:
        voice_totals = usage_service.totals(voice_records)
        voice_columns = st.columns(4, gap="medium", border=True)
        voice_columns[0].metric("Chamadas", len(voice_records))
        voice_columns[1].metric("Caracteres", _format_tokens(voice_totals.input_characters))
        voice_columns[2].metric(
            "Tokens de áudio", _format_tokens(voice_totals.audio_output_tokens)
        )
        voice_columns[3].metric("Duração", f"{voice_totals.audio_seconds / 60:.2f} min")
        st.space("small")
        voice_groups = usage_service.grouped_totals(voice_records, "model")
        st.dataframe(
            [
                {
                    "Modelo": model,
                    "Chamadas": values.completed_calls + values.failed_calls,
                    "Caracteres": values.input_characters,
                    "Tokens de entrada": values.input_tokens,
                    "Tokens de áudio": values.audio_output_tokens,
                    "Duração (s)": round(values.audio_seconds, 2),
                    "Custo estimado (USD)": (
                        values.estimated_cost_usd if values.estimated_cost_usd else None
                    ),
                }
                for model, values in voice_groups.items()
            ],
            width="stretch",
            hide_index=True,
        )

with provider_tab:
    selected_provider = st.selectbox(
        "Provedor",
        list(provider_groups),
        key="usage_provider_detail",
    )
    provider_records = [record for record in records if record.provider == selected_provider]
    provider_totals = usage_service.totals(provider_records)
    provider_columns = st.columns(3, gap="medium", border=True)
    provider_columns[0].metric("Chamadas", len(provider_records))
    provider_columns[1].metric("Tokens", _format_tokens(provider_totals.total_tokens))
    provider_columns[2].metric("Custo estimado", _format_cost(provider_totals.estimated_cost_usd))
    agent_groups = usage_service.grouped_totals(provider_records, "agent_type")
    st.dataframe(
        [
            {
                "Agente": _AGENT_LABELS.get(agent, agent),
                "Tokens": values.total_tokens,
                "Concluídas": values.completed_calls,
                "Repetidas": values.repeated_calls,
                "Falhas": values.failed_calls,
            }
            for agent, values in agent_groups.items()
        ],
        width="stretch",
        hide_index=True,
    )

with calls_tab:
    st.dataframe(
        [
            {
                "Data": record.created_at,
                "Projeto": project_titles.get(record.project_id, record.project_id),
                "Provedor": record.provider,
                "Modelo": record.model,
                "Modalidade": "Voz" if record.modality == "speech" else "Texto/multimodal",
                "Agente": _AGENT_LABELS.get(record.agent_type, record.agent_type),
                "Etapa": _STAGE_LABELS.get(record.stage, record.stage),
                "Status": "Concluída" if record.status == "completed" else "Falha",
                "Tentativa": (
                    "Fallback/repetição" if record.attempt_type == "fallback" else "Principal"
                ),
                "Repetida": "Sim" if record.sequence > 1 else "Não",
                "Entrada": record.input_tokens,
                "Saída": record.output_tokens,
                "Cache lido": record.cache_read_tokens,
                "Cache criado": record.cache_creation_tokens,
                "Raciocínio": record.reasoning_tokens,
                "Total": record.total_tokens,
                "Caracteres TTS": record.input_characters,
                "Tokens de áudio": record.audio_output_tokens,
                "Duração (s)": round(record.audio_seconds, 2),
                "Custo estimado (USD)": (
                    record.estimated_cost_usd if record.pricing_known else None
                ),
                "Origem": record.usage_source,
            }
            for record in reversed(records)
        ],
        width="stretch",
        hide_index=True,
    )
