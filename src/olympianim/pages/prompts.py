"""Prompt editor page for agent templates."""

from __future__ import annotations

import streamlit as st

from olympianim.prompts.service import PromptService, PromptWithVersion
from olympianim.ui import state


def _session_prompt_values() -> dict[str, object]:
    """Build render variables from current Streamlit session state."""
    return {
        "problem_statement": state.get(state.KEY_PROBLEM_STATEMENT, ""),
        "problem_image_description": "",
        "teacher_instructions": state.get(state.KEY_TEACHER_INSTRUCTIONS, ""),
        "solution_basis": "",
        "approved_plan": "",
        "manim_code": "",
        "render_error": "",
        "voiceover_requirements": "",
        "video_mode": "presentation",
        "transcript": "Leia o enunciado com clareza.",
        "language": state.get(state.KEY_VOICE_LANGUAGE, "Português (Brasil)"),
    }


def _prompt_label(prompt: PromptWithVersion) -> str:
    suffix = "padrão" if prompt.prompt.is_default else "personalizado"
    return f"{prompt.prompt.name} | v{prompt.latest_version.version} | {suffix}"


st.title("Prompts")
st.caption("Gerencie os templates de prompts dos agentes.")

service = PromptService()
service.ensure_default_prompts()

agents = list(service.list_agents())
agent_labels = {agent.display_name: agent.agent_type for agent in agents}
selected_agent_label = st.selectbox("Agente", options=tuple(agent_labels))
selected_agent = agent_labels[selected_agent_label]
agent_spec = next(agent for agent in agents if agent.agent_type == selected_agent)

st.caption(agent_spec.description)

with st.expander("Variáveis disponíveis para este agente", expanded=True):
    st.code("\n".join(f"{{{variable}}}" for variable in agent_spec.variables), language=None)

prompts = service.list_prompts(selected_agent)
if not prompts:
    st.info("Nenhum prompt cadastrado para este agente.")
    st.stop()

prompt_by_label = {_prompt_label(prompt): prompt for prompt in prompts}
selected_prompt_label = st.selectbox("Prompt", options=tuple(prompt_by_label))
selected_prompt = prompt_by_label[selected_prompt_label]
editor_key = f"prompt_editor_{selected_prompt.prompt.id}_{selected_prompt.latest_version.version}"

st.text_input("Nome", value=selected_prompt.prompt.name, disabled=True)
st.text_area(
    "Template",
    value=selected_prompt.latest_version.template_text,
    height=420,
    key=editor_key,
)

template_text = st.session_state[editor_key]
validation = service.validate(selected_prompt.prompt.agent_type, template_text)

if validation.used_variables:
    st.caption("Variáveis usadas: " + ", ".join(f"{{{v}}}" for v in validation.used_variables))
else:
    st.caption("Nenhuma variável usada no template.")

if validation.unknown_variables:
    st.error(
        "Variável não reconhecida: " + ", ".join(f"{{{v}}}" for v in validation.unknown_variables)
    )

col_save, col_restore, col_duplicate = st.columns(3)

with col_save:
    if st.button(
        "Salvar nova versão",
        type="primary",
        icon=":material/save:",
        width="stretch",
    ):
        version, result = service.save_prompt_version(selected_prompt.prompt.id, template_text)
        if version is None:
            st.error(
                "Não foi possível salvar. Variáveis desconhecidas: "
                + ", ".join(f"{{{v}}}" for v in result.unknown_variables)
            )
        else:
            st.success(f"Versão {version.version} salva.")
            st.rerun()

with col_restore:
    if st.button(
        "Restaurar padrão",
        icon=":material/restore:",
        width="stretch",
    ):
        version = service.restore_default_prompt(selected_prompt.prompt.id)
        st.success(f"Padrão restaurado como versão {version.version}.")
        st.rerun()

with col_duplicate:
    duplicate_name = st.text_input(
        "Nome da cópia",
        value=f"{selected_prompt.prompt.name} - cópia",
        label_visibility="collapsed",
    )
    if st.button("Duplicar", icon=":material/content_copy:", width="stretch"):
        duplicate = service.duplicate_prompt(selected_prompt.prompt.id, duplicate_name)
        st.success(f"Prompt duplicado: {duplicate.prompt.name}.")
        st.rerun()

current_project_id = state.get(state.KEY_CURRENT_PROJECT_ID, "")
if current_project_id:
    st.divider()
    st.subheader("Snapshot do projeto atual")
    st.caption(
        "Salva a versão renderizada do prompt selecionado para reprodutibilidade do projeto aberto."
    )
    if st.button("Salvar snapshot do prompt usado", icon=":material/archive:"):
        snapshot = service.save_project_prompt_snapshot(
            current_project_id,
            agent_type=selected_prompt.prompt.agent_type,
            prompt_id=selected_prompt.prompt.id,
            values=_session_prompt_values(),
        )
        st.success(f"Snapshot salvo para a versão {snapshot.prompt_version}.")
