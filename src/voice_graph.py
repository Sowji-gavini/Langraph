import os
import tempfile
import operator

from typing import (
    List,
    Optional,
    TypedDict,
    Annotated,
)

from groq import Groq

from gtts import gTTS

from langchain_core.documents import Document

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
)

from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)

from langchain_community.vectorstores import FAISS

from langchain_groq import ChatGroq

from langgraph.graph import (
    StateGraph,
    START,
    END,
)

from src.config import config
from src.vector_store import get_retriever


# ─────────────────────────────────────────────────────
# State
# ─────────────────────────────────────────────────────

class VoiceState(TypedDict):

    audio_bytes: Optional[bytes]

    transcript: str

    chat_history: Annotated[
        List[BaseMessage],
        operator.add
    ]

    retrieved_docs: List[Document]

    answer: str

    audio_response: Optional[bytes]


# ─────────────────────────────────────────────────────
# Transcribe Node
# ─────────────────────────────────────────────────────

def make_transcribe_node(
    client: Groq
):

    def transcribe(
        state: VoiceState
    ):

        audio_bytes = state.get(
            "audio_bytes"
        )

        if not audio_bytes:

            return {
                "transcript": ""
            }

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".wav"
        ) as tmp:

            tmp.write(audio_bytes)

            tmp_path = tmp.name

        try:

            with open(
                tmp_path,
                "rb"
            ) as audio_file:

                transcription = (
                    client.audio.transcriptions.create(
                        file=audio_file,
                        model=config.WHISPER_MODEL,
                        language="en",
                    )
                )

            transcript = (
                transcription.text.strip()
            )

        finally:

            os.remove(tmp_path)

        return {
            "transcript": transcript
        }

    return transcribe


# ─────────────────────────────────────────────────────
# Retrieve Node
# ─────────────────────────────────────────────────────

def make_voice_retrieve_node(
    vector_store: FAISS
):

    retriever = get_retriever(
        vector_store
    )

    def retrieve(
        state: VoiceState
    ):

        transcript = state.get(
            "transcript",
            ""
        )

        docs = retriever.invoke(
            transcript
        )

        return {
            "retrieved_docs": docs
        }

    return retrieve


# ─────────────────────────────────────────────────────
# Generate Node
# ─────────────────────────────────────────────────────

def make_voice_generate_node(
    llm: ChatGroq
):

    system_prompt = """
You are a helpful AI voice assistant.

Answer ONLY from the given context.

Context:
{context}

Rules:
- Keep answers short.
- Speak naturally.
- If answer not found say:
"I couldn't find that information in the uploaded document."
"""

    prompt = ChatPromptTemplate.from_messages([

        ("system", system_prompt),

        MessagesPlaceholder(
            variable_name="chat_history"
        ),

        ("human", "{question}")
    ])

    chain = prompt | llm

    def generate(
        state: VoiceState
    ):

        context = "\n\n".join([

            doc.page_content
            for doc in state["retrieved_docs"]

        ])

        response = chain.invoke({

            "context": context,

            "chat_history": state["chat_history"],

            "question": state["transcript"],
        })

        return {
            "answer": response.content
        }

    return generate


# ─────────────────────────────────────────────────────
# TTS Node
# ─────────────────────────────────────────────────────

def make_synthesize_node():

    def synthesize(
        state: VoiceState
    ):

        answer = state.get(
            "answer",
            ""
        )

        if not answer:

            return {
                "audio_response": None
            }

        tts = gTTS(
            text=answer,
            lang="en"
        )

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp3"
        ) as tmp:

            temp_audio = tmp.name

        tts.save(temp_audio)

        with open(
            temp_audio,
            "rb"
        ) as f:

            audio_bytes = f.read()

        os.remove(temp_audio)

        return {
            "audio_response": audio_bytes
        }

    return synthesize


# ─────────────────────────────────────────────────────
# Build Graph
# ─────────────────────────────────────────────────────

def build_voice_graph(
    vector_store: FAISS
):

    client = Groq(
        api_key=config.GROQ_API_KEY
    )

    llm = ChatGroq(

        groq_api_key=config.GROQ_API_KEY,

        model_name=config.GROQ_MODEL,

        temperature=0.3,
    )

    transcribe_node = (
        make_transcribe_node(client)
    )

    retrieve_node = (
        make_voice_retrieve_node(
            vector_store
        )
    )

    generate_node = (
        make_voice_generate_node(
            llm
        )
    )

    synthesize_node = (
        make_synthesize_node()
    )

    graph = StateGraph(
        VoiceState
    )

    graph.add_node(
        "transcribe",
        transcribe_node
    )

    graph.add_node(
        "retrieve",
        retrieve_node
    )

    graph.add_node(
        "generate",
        generate_node
    )

    graph.add_node(
        "synthesize",
        synthesize_node
    )

    graph.add_edge(
        START,
        "transcribe"
    )

    graph.add_edge(
        "transcribe",
        "retrieve"
    )

    graph.add_edge(
        "retrieve",
        "generate"
    )

    graph.add_edge(
        "generate",
        "synthesize"
    )

    graph.add_edge(
        "synthesize",
        END
    )

    return graph.compile()


# ─────────────────────────────────────────────────────
# Run Query
# ─────────────────────────────────────────────────────

def run_voice_query(
    app,
    audio_bytes: bytes,
    chat_history: List[BaseMessage],
):

    initial_state = {

        "audio_bytes": audio_bytes,

        "transcript": "",

        "chat_history": chat_history,

        "retrieved_docs": [],

        "answer": "",

        "audio_response": None,
    }

    result = app.invoke(
        initial_state
    )

    transcript = result["transcript"]

    answer = result["answer"]

    audio_response = result["audio_response"]

    updated_history = chat_history + [

        HumanMessage(
            content=transcript
        ),

        AIMessage(
            content=answer
        ),
    ]

    return (
        transcript,
        answer,
        audio_response,
        updated_history,
    )