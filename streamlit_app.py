import streamlit as st
import csv
from pathlib import Path
from collections import Counter, defaultdict
from pypdf import PdfReader  # Correction de la lecture PDF

from app.router import AgentRouter



# Paths


ADMIN_DIR = Path("data/admin")
CONTACTS_PATH = ADMIN_DIR / "contacts.csv"
VOTES_PATH = ADMIN_DIR / "votes.csv"
USAGE_PATH = ADMIN_DIR / "usage.csv"

ADMIN_DIR.mkdir(parents=True, exist_ok=True)



# Utils

def valid_email(email):
    return "@" in email and "." in email


def valid_phone(phone):
    return phone.isdigit() or phone == ""


def append_csv(path, header, row):
    exists = path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow(header)
        writer.writerow(row)


def init_session():
    if "router" not in st.session_state:
        st.session_state.router = AgentRouter()
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "uploaded_doc" not in st.session_state:
        st.session_state.uploaded_doc = None
    if "votes" not in st.session_state:
        st.session_state.votes = {}



# APP


def main():
    st.set_page_config("ESILV Smart Assistant", "🎓", layout="centered")
    init_session()

    
    page = st.sidebar.radio("Navigation", ["💬 Chat", "📊 Admin"])

    # FORMULAIRE DE CONTACT 
    st.sidebar.markdown("## 📩 Contact ")
    with st.sidebar.form("contact_form"):
        name = st.text_input("Nom *")
        email = st.text_input("Email *")
        phone = st.text_input("Téléphone")
        subject = st.selectbox(
            "Sujet",
            [
                "Admissions",
                "Vie étudiante",
                "Académique",
                "Administratif",
                "Autre",
            ],
        )
        comment = st.text_area("Commentaire")

        submitted = st.form_submit_button("Envoyer")

        if submitted:
            if not name or not email:
                st.warning("Nom et email requis.")
            elif not valid_email(email):
                st.warning("Email invalide.")
            elif not valid_phone(phone):
                st.warning("Téléphone invalide.")
            else:
                append_csv(
                    CONTACTS_PATH,
                    ["name", "email", "phone", "subject", "comment"],
                    [name, email, phone, subject, comment],
                )
                st.success("Message envoyé.")

    
    # PAGE CHAT
    
    if page == "💬 Chat":
        st.title("🎓 ESILV Smart Assistant")
        st.caption("Assistant multi-agents avec RAG institutionnel.")

        # Historique 
        for i, msg in enumerate(st.session_state.messages):
            with st.chat_message(msg["role"]):
                if msg["role"] == "assistant":
                    st.markdown(f"**{msg['agent']}**")
                    st.markdown(msg["content"])

                    c1, c2, _ = st.columns([1, 1, 6])
                    if c1.button("👍", key=f"up_{i}"):
                        st.session_state.votes[i] = "up"
                        append_csv(
                            VOTES_PATH,
                            ["msg_id", "agent", "vote"],
                            [i, msg["agent"], "up"],
                        )
                    if c2.button("👎", key=f"down_{i}"):
                        st.session_state.votes[i] = "down"
                        append_csv(
                            VOTES_PATH,
                            ["msg_id", "agent", "vote"],
                            [i, msg["agent"], "down"],
                        )
                else:
                    st.markdown(msg["content"])

        # Upload document 
        with st.expander("📎 Ajouter un document", expanded=False):
            file = st.file_uploader("", type=["txt", "pdf"], label_visibility="collapsed")
            if file:
                if file.name.endswith(".pdf"):
                    try:
                        # Utilisation de pypdf pour extraire le texte réel 
                        pdf_reader = PdfReader(file)
                        text_content = ""
                        for page_num in range(len(pdf_reader.pages)):
                            page_text = pdf_reader.pages[page_num].extract_text()
                            if page_text:
                                text_content += page_text + "\n"
                        st.session_state.uploaded_doc = text_content
                        st.success(f"PDF '{file.name}' lu avec succès")
                    except Exception as e:
                        st.error(f"Erreur lors de la lecture du PDF : {e}")
                else:
                    # Pour les fichiers TXT
                    st.session_state.uploaded_doc = file.read().decode(
                        "utf-8", errors="ignore"
                    )
                    st.success(f"{file.name} chargé")

        # Chat input
        user_input = st.chat_input("Pose ta question…")

        # Afficher la question immédiatement
        if user_input:
            st.session_state.messages.append(
                {
                    "role": "user",
                    "content": user_input,
                    "uploaded_doc": st.session_state.uploaded_doc,
                }
            )
            st.session_state.pending_question = user_input
            st.rerun()

        # Calculer la réponse
        if "pending_question" in st.session_state:
            with st.chat_message("assistant"):
                with st.spinner("Réflexion en cours..."):
                    agent_response = st.session_state.router.handle(
                        st.session_state.pending_question,
                        history=st.session_state.messages,
                    )

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": agent_response.answer,
                    "agent": agent_response.agent_name,
                }
            )

            append_csv(
                USAGE_PATH,
                ["agent", "question"],
                [
                    agent_response.agent_name,
                    st.session_state.pending_question,
                ],
            )

            del st.session_state.pending_question
            st.rerun()


    # PAGE ADMIN 

    else:
        st.title("📊 Admin – Statistiques")

        if USAGE_PATH.exists():
            with open(USAGE_PATH, encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
        else:
            rows = []

        st.metric("Nombre total de questions", len(rows))

        count_agents = Counter(r["agent"] for r in rows)
        st.subheader("Questions par agent")
        for agent, n in count_agents.items():
            st.write(f"**{agent}** : {n}")

        stats = defaultdict(lambda: {"up": 0, "down": 0})
        if VOTES_PATH.exists():
            with open(VOTES_PATH, encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    stats[r["agent"]][r["vote"]] += 1

        st.subheader("Qualité des réponses")
        for agent, s in stats.items():
            total = s["up"] + s["down"]
            if total == 0:
                continue
            st.write(
                f"**{agent}** → 👍 {round(s['up']/total*100,1)}% | "
                f"👎 {round(s['down']/total*100,1)}%"
            )


if __name__ == "__main__":
    main()