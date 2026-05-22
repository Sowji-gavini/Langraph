from typing import List, TypedDict, Annotated
import operator

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

class RAGState(TypedDict):

    question: str

    chat_history: Annotated[
        List[BaseMessage],
        operator.add
    ]

    retrieved_docs: List[Document]

    answer: str


# ─────────────────────────────────────────────────────
# Retrieve Node
# ─────────────────────────────────────────────────────

def make_retrieve_node(
    vector_store: FAISS
):

    retriever = get_retriever(
        vector_store
    )

    def retrieve(
        state: RAGState
    ):

        docs = retriever.invoke(
            state["question"]
        )

        return {
            "retrieved_docs": docs
        }

    return retrieve


# ─────────────────────────────────────────────────────
# Generate Node
# ─────────────────────────────────────────────────────

def make_generate_node(
    llm: ChatGroq
):

    system_prompt = """
You are a helpful AI assistant.

Answer ONLY from the provided context.

Context:
{context}

Rules:
- If answer not found, say:
"I couldn't find that information in the uploaded document."
- Keep answers concise.
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
        state: RAGState
    ):

        context = "\n\n".join([

            doc.page_content
            for doc in state["retrieved_docs"]

        ])

        response = chain.invoke({

            "context": context,

            "chat_history": state["chat_history"],

            "question": state["question"],
        })

        return {
            "answer": response.content
        }

    return generate


# ─────────────────────────────────────────────────────
# Build Graph
# ─────────────────────────────────────────────────────

def build_rag_graph(
    vector_store: FAISS
):

    llm = ChatGroq(

        groq_api_key=config.GROQ_API_KEY,

        model_name=config.GROQ_MODEL,

        temperature=0.3,
    )

    retrieve_node = make_retrieve_node(
        vector_store
    )

    generate_node = make_generate_node(
        llm
    )

    graph = StateGraph(
        RAGState
    )

    graph.add_node(
        "retrieve",
        retrieve_node
    )

    graph.add_node(
        "generate",
        generate_node
    )

    graph.add_edge(
        START,
        "retrieve"
    )

    graph.add_edge(
        "retrieve",
        "generate"
    )

    graph.add_edge(
        "generate",
        END
    )

    return graph.compile()


# ─────────────────────────────────────────────────────
# Run Query
# ─────────────────────────────────────────────────────

def run_rag_query(
    app,
    question: str,
    chat_history: List[BaseMessage],
):

    initial_state = {

        "question": question,

        "chat_history": chat_history,

        "retrieved_docs": [],

        "answer": "",
    }

    result = app.invoke(
        initial_state
    )

    answer = result["answer"]

    updated_history = chat_history + [

        HumanMessage(
            content=question
        ),

        AIMessage(
            content=answer
        ),
    ]

    return answer, updated_history