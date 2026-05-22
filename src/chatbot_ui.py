"""
src/chatbot_ui.py
──────────────────
Streamlit UI for the text chatbot mode.
"""

import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage

from src.rag_graph import build_rag_graph, run_rag_query


def render_chatbot_ui(vector_store):
    """
    Render the chatbot interface.

    Args:
        vector_store: FAISS vector store already built from the uploaded doc
    """

    # ── Session state initialisation ─────────────────────────────────────────
    if "chat_rag_app" not in st.session_state:
        with st.spinner("🔧 Building RAG pipeline..."):
            st.session_state.chat_rag_app = build_rag_graph(vector_store)

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []

    # ── Header ───────────────────────────────────────────────────────────────
    st.markdown("""
    <div style='background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
                padding: 1.5rem 2rem; border-radius: 12px; margin-bottom: 1.5rem;
                border: 1px solid rgba(100, 160, 255, 0.2);'>
        <h2 style='margin:0; color:#64a0ff; font-family: monospace;'>
            💬 Document Chatbot
        </h2>
        <p style='margin:0.5rem 0 0; color:#8899aa; font-size:0.9rem;'>
            Ask anything about your uploaded document
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Chat controls ────────────────────────────────────────────────────────
    col1, col2 = st.columns([6, 1])
    with col2:
        if st.button("🗑️ Clear", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.chat_messages = []
            st.rerun()

    # ── Chat display ─────────────────────────────────────────────────────────
    chat_container = st.container(height=420)
    with chat_container:
        if not st.session_state.chat_messages:
            st.markdown("""
            <div style='text-align:center; color:#556677; padding: 3rem 0;'>
                <div style='font-size:2.5rem;'>🤖</div>
                <p>Ask me anything about your document!</p>
                <p style='font-size:0.8rem;'>I have read and indexed the entire file for you.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            for msg in st.session_state.chat_messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

    # ── Input ────────────────────────────────────────────────────────────────
    user_input = st.chat_input("Ask a question about the document...")

    if user_input:
        # Add user message immediately
        st.session_state.chat_messages.append({
            "role": "user",
            "content": user_input,
        })

        with st.spinner("🔍 Searching document and generating answer..."):
            answer, updated_history = run_rag_query(
                app=st.session_state.chat_rag_app,
                question=user_input,
                chat_history=st.session_state.chat_history,
            )

        st.session_state.chat_history = updated_history
        st.session_state.chat_messages.append({
            "role": "assistant",
            "content": answer,
        })
        st.rerun()

    # ── Stats ────────────────────────────────────────────────────────────────
    if st.session_state.chat_messages:
        n_turns = len(st.session_state.chat_messages) // 2
        st.caption(f"📊 Conversation turns: **{n_turns}**")