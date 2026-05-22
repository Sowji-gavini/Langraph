"""
src/voicebot_ui.py
──────────────────
Streamlit UI for Voice RAG chatbot.

Features:
- Mic recording
- Speech-to-text
- RAG retrieval
- Voice response playback
- Chat history
"""

import base64

import streamlit as st

from streamlit_mic_recorder import (
    mic_recorder
)

from src.voice_graph import (
    build_voice_graph,
    run_voice_query,
)


# ─────────────────────────────────────────────────────────────────────────────
# Audio Player
# ─────────────────────────────────────────────────────────────────────────────

def autoplay_audio(
    audio_bytes: bytes
):
    """
    Auto-play generated audio response.
    """

    b64 = base64.b64encode(
        audio_bytes
    ).decode()

    audio_html = f"""
    <audio autoplay controls style="width:100%;">
        <source
            src="data:audio/mp3;base64,{b64}"
            type="audio/mp3"
        >
    </audio>
    """

    st.markdown(
        audio_html,
        unsafe_allow_html=True
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main Voice UI
# ─────────────────────────────────────────────────────────────────────────────

def render_voicebot_ui(
    vector_store
):
    """
    Render VoiceBot interface.
    """

    # ─────────────────────────────────────────────────────────────────────────
    # Session State
    # ─────────────────────────────────────────────────────────────────────────

    if "voice_rag_app" not in st.session_state:

        with st.spinner(
            "🔧 Building Voice RAG Pipeline..."
        ):

            st.session_state.voice_rag_app = (
                build_voice_graph(
                    vector_store
                )
            )

    if "voice_history" not in st.session_state:
        st.session_state.voice_history = []

    if "voice_messages" not in st.session_state:
        st.session_state.voice_messages = []

    # ─────────────────────────────────────────────────────────────────────────
    # Header
    # ─────────────────────────────────────────────────────────────────────────

    st.markdown(
        """
        <div style='
            background: linear-gradient(
                135deg,
                #16213e 0%,
                #0f3460 100%
            );
            padding: 1.5rem;
            border-radius: 15px;
            margin-bottom: 1rem;
        '>

        <h2 style='
            color:white;
            margin:0;
        '>
            🎙️ Voice RAG Assistant
        </h2>

        <p style='
            color:#d0d0d0;
            margin-top:0.5rem;
        '>
            Ask questions using your voice
        </p>

        </div>
        """,
        unsafe_allow_html=True,
    )

    # ─────────────────────────────────────────────────────────────────────────
    # Clear Button
    # ─────────────────────────────────────────────────────────────────────────

    col1, col2 = st.columns([6, 1])

    with col2:

        if st.button(
            "🗑️ Clear",
            use_container_width=True
        ):

            st.session_state.voice_history = []

            st.session_state.voice_messages = []

            st.rerun()

    # ─────────────────────────────────────────────────────────────────────────
    # Recorder
    # ─────────────────────────────────────────────────────────────────────────

    st.markdown(
        "### 🎤 Record Your Question"
    )

    audio_data = mic_recorder(
        start_prompt="🎙️ Start Recording",

        stop_prompt="⏹️ Stop Recording",

        key="voice_recorder",

        format="wav",

        use_container_width=True,
    )

    # ─────────────────────────────────────────────────────────────────────────
    # Process Audio
    # ─────────────────────────────────────────────────────────────────────────

    if (
        audio_data
        and audio_data.get("bytes")
    ):

        raw_audio = audio_data["bytes"]

        with st.spinner(
            "🎧 Processing Voice Query..."
        ):

            transcript, answer, audio_response, updated_history = (
                run_voice_query(
                    app=st.session_state.voice_rag_app,

                    audio_bytes=raw_audio,

                    chat_history=st.session_state.voice_history,
                )
            )

        # Store conversation
        if transcript:

            st.session_state.voice_history = (
                updated_history
            )

            st.session_state.voice_messages.append({

                "role": "user",

                "transcript": transcript,

                "audio": raw_audio,
            })

            st.session_state.voice_messages.append({

                "role": "assistant",

                "answer": answer,

                "audio": audio_response,
            })

    # ─────────────────────────────────────────────────────────────────────────
    # Chat History
    # ─────────────────────────────────────────────────────────────────────────

    st.markdown("---")

    st.markdown(
        "### 💬 Conversation"
    )

    if not st.session_state.voice_messages:

        st.info(
            "Record your voice to start chatting."
        )

    else:

        messages = (
            st.session_state.voice_messages
        )

        for i in range(
            len(messages) - 1,
            -1,
            -2
        ):

            assistant_msg = messages[i]

            user_msg = (
                messages[i - 1]
                if i > 0
                else None
            )

            # User Message
            if user_msg:

                with st.chat_message(
                    "user"
                ):

                    st.markdown(
                        f"🗣️ {user_msg['transcript']}"
                    )

            # Assistant Message
            with st.chat_message(
                "assistant"
            ):

                st.markdown(
                    assistant_msg["answer"]
                )

                if assistant_msg.get(
                    "audio"
                ):

                    autoplay_audio(
                        assistant_msg["audio"]
                    )

    # ─────────────────────────────────────────────────────────────────────────
    # Info Section
    # ─────────────────────────────────────────────────────────────────────────

    with st.expander(
        "ℹ️ Voice Pipeline"
    ):

        st.markdown(
            """
### Voice RAG Flow

```text
🎤 User Voice
      ↓
🧠 Groq Whisper
      ↓
📚 FAISS Retriever
      ↓
🤖 Groq LLM
      ↓
🔊 gTTS Voice Output
"""
        )