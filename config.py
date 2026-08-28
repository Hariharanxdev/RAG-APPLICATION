import os
from dotenv import load_dotenv

load_dotenv()

APP_NAME = "RAG Assistant"

TOP_K = 5

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

SIMILARITY_THRESHOLD = 0.45

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

GROQ_MODEL = "openai/gpt-oss-20b"