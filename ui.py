import time
from pathlib import Path

import streamlit as st

from document_loader import load_document
from chunker import create_chunks
from retriever import Retriever
from chatbot import Chatbot
from logger import log_question, log_retrieval, log_answer


DATA_DIR = Path("data")


# =========================================
# LOAD ALL DOCUMENTS
# =========================================

def load_all_documents():
    """Load all supported documents from the data folder."""

    all_chunks = []
    total_pages = 0

    supported_extensions = {
        ".pdf",
        ".docx",
        ".txt"
    }

    all_files = [
        file
        for file in DATA_DIR.iterdir()
        if file.is_file()
    ]

    # ---------------------------------
    # Unsupported files
    # ---------------------------------

    unsupported_files = [
        file
        for file in all_files
        if file.suffix.lower()
        not in supported_extensions
    ]

    if unsupported_files:
        for file_path in unsupported_files:
            print(
                f"Unsupported file: "
                f"{file_path.name}"
            )

    # ---------------------------------
    # Supported files
    # ---------------------------------

    files = [
        file
        for file in all_files
        if file.suffix.lower()
        in supported_extensions
    ]

    if not files:
        raise FileNotFoundError(
            "No PDF, DOCX, or TXT files "
            "found in the data folder."
        )

    valid_files = []

    # ---------------------------------
    # Load documents
    # ---------------------------------

    for file_path in files:

        try:

            pages = load_document(
                file_path
            )

            chunks = create_chunks(
                pages,
                file_path.name
            )

            all_chunks.extend(chunks)

            total_pages += len(pages)

            valid_files.append(
                file_path
            )

            print(
                f"Loaded: {file_path.name} | "
                f"Pages: {len(pages)} | "
                f"Chunks: {len(chunks)}"
            )

        except Exception as error:

            print(
                f"Error loading "
                f"{file_path.name}: {error}"
            )

            print(
                f"Skipping "
                f"{file_path.name}..."
            )

            continue

    # ---------------------------------
    # Average chunk size
    # ---------------------------------

    if all_chunks:

        total_characters = sum(
            len(chunk["text"])
            for chunk in all_chunks
        )

        average_chunk_size = (
            total_characters /
            len(all_chunks)
        )

    else:

        average_chunk_size = 0

    return (
        all_chunks,
        valid_files,
        total_pages,
        average_chunk_size
    )


# =========================================
# CONFIDENCE
# =========================================

def get_confidence(results):
    """
    Calculate confidence using the
    Hybrid Search score.
    """

    if not results:
        return "Low"

    best_score = max(
        result.get("hybrid_score", 0.0)
        for result in results
    )

    if best_score >= 0.60:
        return "High"

    if best_score >= 0.40:
        return "Medium"

    return "Low"


# =========================================
# INITIALIZE RAG
# =========================================

@st.cache_resource
def initialize_rag():
    """Initialize the RAG system."""

    (
        chunks,
        files,
        total_pages,
        average_chunk_size
    ) = load_all_documents()

    retriever = Retriever()

    retriever.add_chunks(
        chunks
    )

    chatbot = Chatbot()

    return (
        retriever,
        chatbot,
        len(chunks),
        files,
        total_pages,
        average_chunk_size
    )


# =========================================
# MAIN
# =========================================

def main():

    # ---------------------------------
    # Page configuration
    # ---------------------------------

    st.set_page_config(
        page_title="RAG Assistant",
        page_icon="📚",
        layout="wide"
    )

    # ---------------------------------
    # Header
    # ---------------------------------

    st.title(
        "📚 RAG Assistant"
    )

    st.markdown(
        "Ask questions about your "
        "**PDF, DOCX, and TXT documents**."
    )

    # ---------------------------------
    # Initialize RAG
    # ---------------------------------

    try:

        (
            retriever,
            chatbot,
            total_chunks,
            files,
            total_pages,
            average_chunk_size
        ) = initialize_rag()

    except Exception as error:

        st.error(
            f"Unable to initialize the "
            f"RAG system: {error}"
        )

        st.stop()

    # ---------------------------------
    # Session state
    # ---------------------------------

    if "messages" not in st.session_state:

        st.session_state.messages = []

    # =================================
    # SIDEBAR
    # =================================

    with st.sidebar:

        st.header(
            "📊 Knowledge Base"
        )

        # Documents

        st.metric(
            "Documents",
            len(files)
        )

        # Pages

        st.metric(
            "Pages",
            total_pages
        )

        # Chunks

        st.metric(
            "Total Chunks",
            total_chunks
        )

        # Average chunk size

        st.metric(
            "Avg Chunk Size",
            f"{average_chunk_size:.2f} chars"
        )

        st.divider()

        # ---------------------------------
        # Documents
        # ---------------------------------

        st.subheader(
            "📄 Documents"
        )

        for file in files:

            st.write(
                f"• {file.name}"
            )

        st.divider()

        # ---------------------------------
        # Actions
        # ---------------------------------

        st.subheader(
            "⚙️ Actions"
        )

        # Clear conversation

        if st.button(
            "🧹 Clear Conversation",
            use_container_width=True
        ):

            chatbot.clear_history()

            st.session_state.messages = []

            st.rerun()

        # Re-index

        if st.button(
            "🔄 Re-index Documents",
            use_container_width=True
        ):

            st.cache_resource.clear()

            st.session_state.messages = []

            st.rerun()

        st.divider()

        # ---------------------------------
        # Pipeline
        # ---------------------------------

        st.info(
            """
            **RAG Pipeline**

            Documents → Chunks → Embeddings
            → ChromaDB + BM25
            → Hybrid Search → Top-5
            → Groq

            **Hybrid Search:** 70% Vector + 30% BM25
            """
        )

    # =================================
    # WELCOME MESSAGE
    # =================================

    if not st.session_state.messages:

        st.info(
            "👋 Welcome! Ask a question "
            "about the documents in your "
            "knowledge base."
        )

    # =================================
    # DISPLAY PREVIOUS CONVERSATION
    # =================================

    for message in st.session_state.messages:

        with st.chat_message(
            message["role"]
        ):

            st.markdown(
                message["content"]
            )

            # ---------------------------------
            # Assistant information
            # ---------------------------------

            if message["role"] == "assistant":

                confidence = message.get(
                    "confidence"
                )

                if confidence:

                    st.caption(
                        f"🎯 Confidence: "
                        f"{confidence}"
                    )

                # ---------------------------------
                # Performance
                # ---------------------------------

                metrics = message.get(
                    "metrics"
                )

                if metrics:

                    st.caption(
                        f"⏱️ Retrieval: "
                        f"{metrics['retrieval_time']:.4f}s  |  "
                        f"LLM: "
                        f"{metrics['llm_time']:.4f}s  |  "
                        f"Total: "
                        f"{metrics['total_time']:.4f}s"
                    )

                # ---------------------------------
                # Sources
                # ---------------------------------

                sources = message.get(
                    "sources",
                    []
                )

                if sources:

                    with st.expander(
                        "📖 View Sources"
                    ):

                        for source in sources:

                            st.markdown(
                                f"""
                                **📄 {source['source']}**

                                Page: `{source['page']}`  
                                Chunk: `{source['chunk_id']}`  

                                🔵 Vector Similarity: `{source.get('similarity', 0):.4f}`  
                                🟢 Keyword Score: `{source.get('keyword_score', 0):.4f}`  
                                🟣 Hybrid Score: `{source.get('hybrid_score', 0):.4f}`
                                """
                            )

                            st.divider()

    # =================================
    # CHAT INPUT
    # =================================

    question = st.chat_input(
        "Ask a question about your documents..."
    )

    if not question:

        return

    # =================================
    # ADD USER MESSAGE
    # =================================

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):

        st.markdown(question)

    # =================================
    # TOTAL TIMER
    # =================================

    total_start_time = time.perf_counter()

    # =================================
    # RETRIEVAL QUERY
    # =================================
    #
    # IMPORTANT:
    # Use the current question for retrieval.
    # The Chatbot handles conversation memory when
    # generating the final answer. Adding the previous
    # question here can dilute the vector/BM25 search
    # and cause correct documents to be missed.
    #
    retrieval_query = question

    # =================================
    # HYBRID RETRIEVAL
    # =================================

    retrieval_start_time = (
        time.perf_counter()
    )

    with st.spinner(
        "Searching your documents..."
    ):

        results = retriever.retrieve(
            retrieval_query,
            top_k=5
        )

    retrieval_end_time = (
        time.perf_counter()
    )

    retrieval_time = (
        retrieval_end_time -
        retrieval_start_time
    )

    # =================================
    # LOG RETRIEVAL
    # =================================

    log_question(
        question
    )

    log_retrieval(
        results
    )

    # =================================
    # GENERATE ANSWER
    # =================================

    llm_start_time = (
        time.perf_counter()
    )

    with st.spinner(
        "Generating answer..."
    ):

        if not results:
            answer = (
                "I couldn't find this information "
                "in the provided documents."
            )
        else:
            answer = chatbot.generate_answer(
                question,
                results
            )

    llm_end_time = (
        time.perf_counter()
    )

    llm_time = (
        llm_end_time -
        llm_start_time
    )

    # =================================
    # TOTAL LATENCY
    # =================================

    total_end_time = (
        time.perf_counter()
    )

    total_time = (
        total_end_time -
        total_start_time
    )

    # =================================
    # LOG ANSWER
    # =================================

    log_answer(
        answer
    )

    # =================================
    # CONFIDENCE
    # =================================

    confidence = get_confidence(
        results
    )

    # =================================
    # SOURCES
    # =================================

    sources = []

    for result in results:

        sources.append(
            {
                "source":
                    result["source"],

                "page":
                    result["page"],

                "chunk_id":
                    result["chunk_id"],

                "similarity":
                    result.get(
                        "similarity",
                        0
                    ),

                "keyword_score":
                    result.get(
                        "keyword_score",
                        0
                    ),

                "hybrid_score":
                    result.get(
                        "hybrid_score",
                        0
                    )
            }
        )

    # =================================
    # PERFORMANCE METRICS
    # =================================

    metrics = {

        "retrieval_time":
            retrieval_time,

        "llm_time":
            llm_time,

        "total_time":
            total_time
    }

    # =================================
    # DISPLAY ASSISTANT RESPONSE
    # =================================

    with st.chat_message(
        "assistant"
    ):

        st.markdown(
            answer
        )

        st.caption(
            f"🎯 Confidence: "
            f"{confidence}"
        )

        # ---------------------------------
        # Performance
        # ---------------------------------

        st.subheader(
            "⏱️ Performance"
        )

        col1, col2, col3 = (
            st.columns(3)
        )

        with col1:

            st.metric(
                "Retrieval Time",
                f"{retrieval_time:.4f} s"
            )

        with col2:

            st.metric(
                "LLM Response Time",
                f"{llm_time:.4f} s"
            )

        with col3:

            st.metric(
                "Total Latency",
                f"{total_time:.4f} s"
            )

        # ---------------------------------
        # Sources
        # ---------------------------------

        if sources:

            with st.expander(
                "📖 View Sources"
            ):

                for source in sources:

                    st.markdown(
                        f"""
                        **📄 {source['source']}**

                        Page: `{source['page']}`  
                        Chunk: `{source['chunk_id']}`  

                        🔵 Vector Similarity: `{source.get('similarity', 0):.4f}`  
                        🟢 Keyword Score: `{source.get('keyword_score', 0):.4f}`  
                        🟣 Hybrid Score: `{source.get('hybrid_score', 0):.4f}`
                        """
                    )

                    st.divider()

    # =================================
    # SAVE ASSISTANT MESSAGE
    # =================================

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "confidence": confidence,
            "sources": sources,
            "metrics": metrics
        }
    )


# =========================================
# ENTRY POINT
# =========================================

if __name__ == "__main__":
    main()