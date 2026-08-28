import logging
from pathlib import Path


LOG_DIR = Path("logs")
LOG_FILE = LOG_DIR / "rag_assistant.log"


# Create logs directory if it doesn't exist
LOG_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# Configure logging
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    encoding="utf-8"
)


logger = logging.getLogger("rag_assistant")


def log_question(question):
    """Log the user's question."""

    logger.info(
        f"QUESTION | {question}"
    )


def log_retrieval(results):
    """Log retrieved chunks and similarity scores."""

    if not results:
        logger.info(
            "RETRIEVAL | No relevant chunks found"
        )
        return

    for result in results:

        logger.info(
            f"RETRIEVAL | "
            f"Source={result['source']} | "
            f"Page={result['page']} | "
            f"Chunk={result['chunk_id']} | "
            f"Similarity={result['similarity']:.4f}"
        )


def log_answer(answer):
    """Log the generated answer."""

    logger.info(
        f"ANSWER | {answer}"
    )