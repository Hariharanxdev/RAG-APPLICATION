from sentence_transformers import SentenceTransformer


MODEL_NAME = "all-MiniLM-L6-v2"


class EmbeddingModel:
    """Generate embeddings using Sentence Transformers."""

    def __init__(self, model_name=MODEL_NAME):
        self.model = SentenceTransformer(model_name)

    def generate_embedding(self, text):
        """Generate an embedding for a single text."""
        return self.model.encode(text).tolist()

    def generate_embeddings(self, texts):
        """Generate embeddings for multiple texts."""
        return self.model.encode(texts).tolist()