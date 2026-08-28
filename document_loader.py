from pathlib import Path
import re

from pypdf import PdfReader
from docx import Document


def clean_text(text):
    """Clean extracted text from documents."""

    # Replace multiple spaces/tabs with a single space
    text = re.sub(r"[ \t]+", " ", text)

    # Remove unnecessary spaces around new lines
    text = re.sub(r" *\n *", "\n", text)

    # Replace 3 or more consecutive new lines with 2
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Remove leading and trailing whitespace
    text = text.strip()

    return text


def load_pdf(file_path):
    """Load text from a PDF file."""

    reader = PdfReader(file_path)

    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""

        text = clean_text(text)

        pages.append({
            "text": text,
            "page": page_number
        })

    return pages


def load_docx(file_path):
    """Load text from a DOCX file."""

    document = Document(file_path)

    text = "\n".join(
        paragraph.text
        for paragraph in document.paragraphs
        if paragraph.text.strip()
    )

    text = clean_text(text)

    return [{
        "text": text,
        "page": 1
    }]


def load_txt(file_path):
    """Load text from a TXT file."""

    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()

    text = clean_text(text)

    return [{
        "text": text,
        "page": 1
    }]


def load_document(file_path):
    """Load a PDF, DOCX, or TXT document."""

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"File not found: {file_path}"
        )

    extension = file_path.suffix.lower()

    if extension == ".pdf":
        return load_pdf(file_path)

    elif extension == ".docx":
        return load_docx(file_path)

    elif extension == ".txt":
        return load_txt(file_path)

    else:
        raise ValueError(
            f"Unsupported file type: {extension}"
        )