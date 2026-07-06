import streamlit as st
import requests
import uuid
# ==============================
# Configuration
# ==============================
API_URL = "http://localhost:8000/chat_llm/memory"
SESSIONS_URL = "http://localhost:8000/sessions"


# ==============================
# FastAPI helpers
# ==============================

def ask_chatbot(message: str, session_id: str) -> dict:
    payload = {"user_message": message, "session_id": session_id}
    try:
        response = requests.post(API_URL, json=payload)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"llm_answer": f"Error Connection to API {e}", "sentiment": "unknown"}


def clear_chat(session_id: str):
    try:
        requests.delete(f"http://localhost:8000/chat/{session_id}")
    except Exception:
        pass


def get_sessions() -> list:
    try:
        r = requests.get(SESSIONS_URL)
        r.raise_for_status()
        return r.json().get("sessions", [])
    except Exception:
        return []


# ==============================
# Streamlit UI
# ==============================

def main():
    st.set_page_config(
        page_title="Chatbot Sentiment Analysis Narendra Modi",
        page_icon="💬",
        layout="wide",
    )
    st.title("💬 Sentiment Analysis — Prime Minister Narendra Modi")

    # Init session_state
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "session_id" not in st.session_state:
        st.session_state["session_id"] = str(uuid.uuid4())
    if "session_titles" not in st.session_state:
        st.session_state["session_titles"] = {}  # session_id -> title

    # ==============================
    # Sidebar: session management
    # ==============================
    with st.sidebar:
        st.header("Chat Sessions")
        if st.button("+ New Chat", key="new_chat", use_container_width=True):
            new_id = str(uuid.uuid4())
            st.session_state["session_id"] = new_id
            st.session_state["messages"] = []
            st.rerun()

        st.divider()

        # Fetch sessions from DB
        all_sessions = get_sessions()

        # Merge with locally-known titles (before DB has them)
        for s in all_sessions:
            sid = s["session_id"]
            st.session_state["session_titles"][sid] = s.get("title", "Untitled")

        # Display session list
        for s in all_sessions:
            sid = s["session_id"]
            title = st.session_state["session_titles"].get(sid, s.get("title", "Untitled"))
            col_sel, col_del = st.columns([4, 1])

            with col_sel:
                is_active = sid == st.session_state["session_id"]
                btn_type = "primary" if is_active else "secondary"
                if st.button(
                    title,
                    key=f"sess_{sid}",
                    use_container_width=True,
                    type=btn_type,
                ):
                    if sid != st.session_state["session_id"]:
                        st.session_state["session_id"] = sid
                        # Load existing messages from DB
                        try:
                            r = requests.get(f"http://localhost:8000/chat/{sid}")
                            r.raise_for_status()
                            data = r.json()
                            st.session_state["messages"] = [
                                (m["role"], m["content"]) for m in data.get("messages", [])
                            ]
                        except Exception:
                            st.session_state["messages"] = []
                        st.rerun()

            with col_del:
                if st.button("🗑", key=f"del_{sid}", help="Delete this session"):
                    clear_chat(sid)
                    st.session_state["session_titles"].pop(sid, None)
                    if sid == st.session_state["session_id"]:
                        st.session_state["messages"] = []
                        st.session_state["session_id"] = str(uuid.uuid4())
                    st.rerun()

        st.divider()
        st.caption(f"Current: `{st.session_state['session_id'][:8]}...`")

    # ==============================
    # Main chat area
    # ==============================

    # Render current title in header
    current_title = st.session_state["session_titles"].get(
        st.session_state["session_id"], "New Chat"
    )
    if current_title == "Untitled":
        current_title = "New Chat"
    st.subheader(current_title)

    # Display chat history
    for role, msg in st.session_state.messages:
        with st.chat_message(role):
            st.write(msg)

    # Chat input
    user_input = st.chat_input("Ask anything about Narendra Modi...")

    if user_input:
        # Save locally
        st.session_state.messages.append(("user", user_input))
        with st.chat_message("user"):
            st.write(user_input)

        # Set title from first message
        if len([m for m in st.session_state.messages if m[0] == "user"]) == 1:
            title = user_input[:50] + ("..." if len(user_input) > 50 else "")
            st.session_state["session_titles"][st.session_state["session_id"]] = title

        # Call API (persisted to MySQL)
        data = ask_chatbot(user_input, st.session_state["session_id"])
        answer = data.get("llm_answer", "No response")
        sentiment = data.get("sentiment", "")

        st.session_state.messages.append(("assistant", answer))
        with st.chat_message("assistant"):
            st.write(answer)
            if sentiment:
                sentiment_label = {0: "Negative 😠", 1: "Neutral 😐", 2: "Positive 😊"}.get(
                    int(sentiment) if str(sentiment).isdigit() else -1, sentiment
                )
                st.caption(f"Sentiment: {sentiment_label}")


if __name__ == "__main__":
    main()
