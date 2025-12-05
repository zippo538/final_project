import streamlit as st
import requests
from src.utils.config import config

# ==============================
# Configuration
# ==============================
API_URL= "http://localhost:8000/chat_llm"

# ==============================
# Send message FastAPI
# ==============================

def ask_chatbot(message:str)-> str : 
    payload = {"user_message" : message}
    
    try : 
        response =  requests.post(API_URL,json=payload)
        response.raise_for_status()
        data = response.json()
        return data
    except Exception as e :
        return f"⚠️Error Connection to API {e}"

# ==============================
# Streamlit UI
# ==============================

def main():
    st.set_page_config(page_title="Chatbot Sentiment Analysis Narenda Modi", page_icon="🤖")
    st.title("🤖 General Sentiment for Prime Minister Narendra Modi (FastAPI + Sreamlit)")

    # Init session state for chat history
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # Display chat history
    for role, msg in st.session_state.messages:
        with st.chat_message(role):
            st.write(msg)

    # Input dari user
    user_input = st.chat_input("Ask Anything...")

    if user_input:
        # Tambahkan ke history
        st.session_state.messages.append(("user", user_input))

        # Tampilkan di UI
        with st.chat_message("user"):
            st.write(user_input)

        # Ambil hasil dari FastAPI
        response_text = ask_chatbot(user_input)

        # Simpan jawaban ke history
        st.session_state.messages.append(("assistant", response_text['llm_answer']))

        # Tampilkan di UI
        with st.chat_message("assistant"):
            st.write(response_text['llm_answer'])


if __name__ == "__main__":
    main()
