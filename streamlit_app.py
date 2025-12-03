# streamlit_app.py

import streamlit as st

from app.router import AgentRouter


def init_session_state():
    if "router" not in st.session_state:
        st.session_state.router = AgentRouter()
    if "messages" not in st.session_state:
        st.session_state.messages = []  # list of dicts: {role, content, agent}


def main():
    st.set_page_config(
        page_title="ESILV Smart Assistant",
        page_icon="🎓",
        layout="centered",
    )

    st.title("🎓 ESILV Smart Assistant – Multi-Agent Chatbot")

    st.markdown(
        """
        Cet assistant utilise plusieurs agents spécialisés (vie étudiante,
        académique, administratif) et un système de RAG sur des documents ESILV.
        """
    )

    init_session_state()

    # Affiche l'historique
    for msg in st.session_state.messages:
        role = "user" if msg["role"] == "user" else "assistant"
        with st.chat_message(role):
            if role == "assistant":
                st.markdown(f"*({msg['agent']})*")
            st.markdown(msg["content"])

    # Zone d'entrée utilisateur
    user_input = st.chat_input("Pose ta question à l'assistant ESILV...")

    if user_input:
        # Ajout du message utilisateur
        st.session_state.messages.append(
            {"role": "user", "content": user_input, "agent": None}
        )

        # Affichage immédiat
        with st.chat_message("user"):
            st.markdown(user_input)

        # Appel du router + agent
        router: AgentRouter = st.session_state.router
        agent_response = router.handle(user_input, history=st.session_state.messages)

        # Ajout réponse agent
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": agent_response.answer,
                "agent": agent_response.agent_name,
            }
        )

        # Affichage réponse
        with st.chat_message("assistant"):
            st.markdown(f"*({agent_response.agent_name})*")
            st.markdown(agent_response.answer)


if __name__ == "__main__":
    main()
