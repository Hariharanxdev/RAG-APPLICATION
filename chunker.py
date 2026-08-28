import json

from config import CHUNK_SIZE, CHUNK_OVERLAP
from pathlib import Path


def chunk_text(text, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    """
    Split text into overlapping chunks.
    """

    if not text:
        return []

    if chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be smaller than chunk_size"
        )

    chunks = []

    start = 0

    while start < len(text):
        end = start + chunk_size

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += chunk_size - chunk_overlap

    return chunks


def create_chunks(pages, source):
    """
    Create chunks from loaded document pages.

    Each chunk contains:
    - text
    - page
    - chunk_id
    - source
    """

    all_chunks = []
    chunk_id = 0

    for page in pages:

        page_text = page["text"]
        page_number = page["page"]

        chunks = chunk_text(page_text)

        for chunk in chunks:

            all_chunks.append({
                "text": chunk,
                "page": page_number,
                "chunk_id": chunk_id,
                "source": source
            })

            chunk_id += 1

    return all_chunks
def get_chunk_statistics(pages, chunks):
    """
    Calculate document processing statistics.
    """

    number_of_pages = len(pages)
    number_of_chunks = len(chunks)

    if number_of_chunks == 0:
        average_chunk_size = 0
    else:
        total_characters = sum(
            len(chunk["text"])
            for chunk in chunks
        )

        average_chunk_size = (
            total_characters / number_of_chunks
        )

    return {
        "number_of_pages": number_of_pages,
        "number_of_chunks": number_of_chunks,
        "average_chunk_size": round(average_chunk_size, 2)
    }

def save_chunks(chunks, output_file="vector_store/chunks.json"):
    """
    Save chunks to a JSON file.
    """

    output_path = Path(output_file)

    # Create parent directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(chunks, file, indent=4, ensure_ascii=False)

    return output_path