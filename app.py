import streamlit as st

from src.config import config
from src.document_processor import process_uploaded_file
from src.vector_store import build_vector_store
from src.chatbot_ui import render_chatbot_ui
from src.voicebot_ui import render_voicebot_ui

st.set_page_config(
    page_title="RAG Bot",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 LangGraph RAG Chatbot + VoiceBot")

# Check API Key
if not config.GROQ_API_KEY:
    st.error("GROQ_API_KEY not found in .env")
    st.stop()

# Upload File
uploaded_file = st.file_uploader(
    "Upload a document",
    type=["pdf", "docx", "txt"]
)

if uploaded_file:

    # Process only once
    if "vector_store" not in st.session_state:

        with st.spinner("Processing document..."):

            chunks = process_uploaded_file(uploaded_file)

            vector_store = build_vector_store(chunks)

            st.session_state.vector_store = vector_store

        st.success("Document processed successfully!")

    # Select mode
    mode = st.radio(
        "Choose Mode",
        ["Chatbot", "VoiceBot"],
        horizontal=True
    )

    vector_store = st.session_state.vector_store

    # Render selected UI
    if mode == "Chatbot":
        render_chatbot_ui(vector_store)

    elif mode == "VoiceBot":
        render_voicebot_ui(vector_store)
        