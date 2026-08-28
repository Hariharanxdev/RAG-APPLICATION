from pathlib import Path

from logger import log_question, log_retrieval, log_answer
from document_loader import load_document
from chunker import create_chunks
from retriever import Retriever
from chatbot import Chatbot


DATA_DIR = Path("data")


def load_all_documents():
    """Load all supported documents from the data folder."""

    all_chunks = []

    supported_extensions = {".pdf", ".docx", ".txt"}

    all_files = [
        file
        for file in DATA_DIR.iterdir()
        if file.is_file()
    ]

    unsupported_files = [
        file
        for file in all_files
        if file.suffix.lower() not in supported_extensions
    ]

    # Display unsupported files
    if unsupported_files:
        print("\nUnsupported files:")

        for file_path in unsupported_files:
            print(
                f"- {file_path.name} "
                f"(unsupported file type)"
            )

    files = [
        file
        for file in all_files
        if file.suffix.lower() in supported_extensions
    ]

    if not files:
        raise FileNotFoundError(
            "No PDF, DOCX, or TXT files found in the data folder."
        )

    for file_path in files:

        print(f"\nLoading: {file_path.name}")

        try:
            pages = load_document(file_path)

            chunks = create_chunks(
                pages,
                file_path.name
            )

            all_chunks.extend(chunks)

            print(f"Pages: {len(pages)}")
            print(f"Chunks: {len(chunks)}")

        except Exception as e:

            print(
                f"Error loading {file_path.name}: {e}"
            )

            print(
                f"Skipping {file_path.name}..."
            )

            continue

    return all_chunks


def calculate_average_chunk_size(chunks):
    """Calculate the average number of characters per chunk."""

    if not chunks:
        return 0

    total_characters = sum(
        len(chunk["text"])
        for chunk in chunks
    )

    return total_characters / len(chunks)


def get_confidence(results):
    """Calculate confidence level from the best similarity score."""

    if not results:
        return "Low"

    best_similarity = max(
        result["similarity"]
        for result in results
    )

    if best_similarity >= 0.60:
        return "High"

    elif best_similarity >= 0.40:
        return "Medium"

    else:
        return "Low"


def main():

    # --------------------------------
    # 1. Load all documents
    # --------------------------------

    chunks = load_all_documents()

    print("\nTotal chunks:", len(chunks))

    # --------------------------------
    # 2. Calculate average chunk size
    # --------------------------------

    average_chunk_size = calculate_average_chunk_size(
        chunks
    )

    print(
        f"Average chunk size: "
        f"{average_chunk_size:.2f} characters"
    )

    # --------------------------------
    # 3. Initialize Retriever
    # --------------------------------

    retriever = Retriever()

    # --------------------------------
    # 4. Store chunks in ChromaDB
    # --------------------------------

    retriever.add_chunks(chunks)

    print("All documents indexed successfully!")

    # --------------------------------
    # 5. Initialize Chatbot
    # --------------------------------

    chatbot = Chatbot()

    # --------------------------------
    # 6. Conversation loop
    # --------------------------------

    while True:

        question = input(
            "\nAsk a question (type 'exit' to quit): "
        )

        # --------------------------------
        # Exit
        # --------------------------------

        if question.lower().strip() == "exit":

            print("Goodbye!")

            break

        # --------------------------------
        # Empty question
        # --------------------------------

        if not question.strip():

            print("Please enter a question.")

            continue

        # --------------------------------
        # Log question
        # --------------------------------

        log_question(question)

        # --------------------------------
        # 7. Create retrieval query
        # --------------------------------

        retrieval_query = question

        if chatbot.conversation_history:

            previous_question = (
                chatbot.conversation_history[-1]["question"]
            )

            previous_answer = (
                chatbot.conversation_history[-1]["answer"]
            )

            retrieval_query = (
                f"Previous question: {previous_question}\n"
                f"Previous answer: {previous_answer}\n"
                f"Current question: {question}"
            )

        # --------------------------------
        # 8. Retrieve relevant chunks
        # --------------------------------

        results = retriever.retrieve(
            retrieval_query,
            top_k=5
        )

        # --------------------------------
        # Log retrieval
        # --------------------------------

        log_retrieval(results)

        # --------------------------------
        # 9. Generate answer
        # --------------------------------

        answer = chatbot.generate_answer(
            question,
            results
        )

        # --------------------------------
        # Log answer
        # --------------------------------

        log_answer(answer)

        # --------------------------------
        # 10. Display answer
        # --------------------------------

        print("\nAssistant:")

        print(answer)

        # --------------------------------
        # 11. Display confidence
        # --------------------------------

        confidence = get_confidence(results)

        print(
            f"\nConfidence: {confidence}"
        )

        # --------------------------------
        # 12. Display sources
        # --------------------------------

        if results:
            print("\nSources:")

            for result in results:
                print(
                    f"- {result['source']} | "
                    f"Page {result['page']} | "
                    f"Chunk {result['chunk_id']} | "
                    f"Vector: {result['similarity']:.4f} | "
                    f"Keyword: {result.get('keyword_score', 0):.4f} | "
                    f"Hybrid: {result.get('hybrid_score', 0):.4f}"
                )


if __name__ == "__main__":
    main()