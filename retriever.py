import re

import chromadb
from rank_bm25 import BM25Okapi

from embeddings import EmbeddingModel
from config import TOP_K, SIMILARITY_THRESHOLD


CHROMA_PATH = "vector_store/chroma_db"
COLLECTION_NAME = "rag_documents"

# Minimum score required for a result
HYBRID_THRESHOLD = 0.30


class Retriever:
    """Hybrid retriever using vector search + BM25 keyword search."""

    def __init__(self):

        # ---------------------------------
        # ChromaDB
        # ---------------------------------

        self.client = chromadb.PersistentClient(
            path=CHROMA_PATH
        )

        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            configuration={
                "hnsw": {
                    "space": "cosine"
                }
            }
        )

        # ---------------------------------
        # Embedding model
        # ---------------------------------

        self.embedding_model = EmbeddingModel()

        # ---------------------------------
        # BM25 data
        # ---------------------------------

        self.chunk_records = {}
        self.bm25 = None
        self.bm25_ids = []

    # =================================
    # TOKENIZER
    # =================================

    def _tokenize(self, text):
        """Convert text into lowercase words."""

        return re.findall(
            r"\b\w+\b",
            text.lower()
        )

    # =================================
    # ADD CHUNKS
    # =================================

    def add_chunks(self, chunks):
        """
        Store chunks in ChromaDB
        and build BM25 index.
        """

        if not chunks:
            return

        # ---------------------------------
        # Documents
        # ---------------------------------

        documents = [
            chunk["text"]
            for chunk in chunks
        ]

        # ---------------------------------
        # Generate embeddings
        # ---------------------------------

        embeddings = (
            self.embedding_model.generate_embeddings(
                documents
            )
        )

        # ---------------------------------
        # IDs
        # ---------------------------------

        ids = [
            f"{chunk['source']}_{chunk['chunk_id']}"
            for chunk in chunks
        ]

        # ---------------------------------
        # Metadata
        # ---------------------------------

        metadatas = [
            {
                "source": chunk["source"],
                "page": chunk["page"],
                "chunk_id": chunk["chunk_id"]
            }
            for chunk in chunks
        ]

        # ---------------------------------
        # Store in ChromaDB
        # ---------------------------------

        self.collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
        )

        # ---------------------------------
        # Store chunks for BM25
        # ---------------------------------

        self.chunk_records = {}

        for chunk in chunks:

            chunk_key = (
                f"{chunk['source']}_{chunk['chunk_id']}"
            )

            self.chunk_records[chunk_key] = {
                "text": chunk["text"],
                "source": chunk["source"],
                "page": chunk["page"],
                "chunk_id": chunk["chunk_id"]
            }

        # ---------------------------------
        # Build BM25 index
        # ---------------------------------

        self._build_bm25_index()

    # =================================
    # BUILD BM25 INDEX
    # =================================

    def _build_bm25_index(self):
        """Create BM25 keyword search index."""

        if not self.chunk_records:

            self.bm25 = None
            self.bm25_ids = []

            return

        self.bm25_ids = list(
            self.chunk_records.keys()
        )

        tokenized_documents = [
            self._tokenize(
                self.chunk_records[chunk_id]["text"]
            )
            for chunk_id in self.bm25_ids
        ]

        self.bm25 = BM25Okapi(
            tokenized_documents
        )

    # =================================
    # NORMALIZE BM25 SCORES
    # =================================

    def _normalize_scores(self, scores):
        """Normalize BM25 scores between 0 and 1."""

        if not scores:
            return []

        minimum = min(scores)
        maximum = max(scores)

        if maximum == minimum:

            if maximum > 0:
                return [
                    1.0
                    for _ in scores
                ]

            return [
                0.0
                for _ in scores
            ]

        return [
            (score - minimum) /
            (maximum - minimum)
            for score in scores
        ]

    # =================================
    # HYBRID RETRIEVAL
    # =================================

    def retrieve(
        self,
        query,
        top_k=TOP_K
    ):
        """
        Hybrid retrieval:

        70% Vector Search
        30% BM25 Keyword Search
        """

        if not query or not query.strip():
            return []

        # =================================
        # 1. VECTOR SEARCH
        # =================================

        collection_count = (
            self.collection.count()
        )

        if collection_count == 0:
            return []

        # ---------------------------------
        # Generate query embedding
        # ---------------------------------

        query_embedding = (
            self.embedding_model.generate_embedding(
                query
            )
        )

        # ---------------------------------
        # Get vector candidates
        # ---------------------------------

        candidate_count = min(
            collection_count,
            max(top_k * 3, top_k)
        )

        vector_results = (
            self.collection.query(
                query_embeddings=[
                    query_embedding
                ],
                n_results=candidate_count
            )
        )

        vector_chunks = {}

        if vector_results["documents"]:

            documents = (
                vector_results["documents"][0]
            )

            metadatas = (
                vector_results["metadatas"][0]
            )

            distances = (
                vector_results["distances"][0]
            )

            for (
                document,
                metadata,
                distance
            ) in zip(
                documents,
                metadatas,
                distances
            ):

                similarity = 1 - distance

                chunk_key = (
                    f"{metadata['source']}_"
                    f"{metadata['chunk_id']}"
                )

                vector_chunks[chunk_key] = {
                    "text": document,
                    "source": metadata["source"],
                    "page": metadata["page"],
                    "chunk_id": metadata["chunk_id"],
                    "similarity": similarity
                }

        # =================================
        # 2. BM25 KEYWORD SEARCH
        # =================================

        keyword_scores = {}

        if self.bm25 is not None:

            query_tokens = (
                self._tokenize(query)
            )

            raw_scores = (
                self.bm25.get_scores(
                    query_tokens
                )
            )

            normalized_scores = (
                self._normalize_scores(
                    list(raw_scores)
                )
            )

            keyword_candidate_count = min(
                len(normalized_scores),
                max(top_k * 3, top_k)
            )

            ranked_indexes = sorted(
                range(
                    len(normalized_scores)
                ),
                key=lambda index:
                    normalized_scores[index],
                reverse=True
            )

            for index in ranked_indexes[
                :keyword_candidate_count
            ]:

                # Ignore zero BM25 matches
                if normalized_scores[index] <= 0:
                    continue

                chunk_key = (
                    self.bm25_ids[index]
                )

                keyword_scores[chunk_key] = (
                    normalized_scores[index]
                )

        # =================================
        # 3. COMBINE RESULTS
        # =================================

        candidate_ids = (
            set(vector_chunks.keys())
            |
            set(keyword_scores.keys())
        )

        hybrid_results = []

        for chunk_key in candidate_ids:

            # ---------------------------------
            # Vector score
            # ---------------------------------

            vector_score = (
                vector_chunks
                .get(chunk_key, {})
                .get("similarity", 0.0)
            )

            # ---------------------------------
            # Keyword score
            # ---------------------------------

            keyword_score = (
                keyword_scores.get(
                    chunk_key,
                    0.0
                )
            )

            # ---------------------------------
            # Hybrid score
            # ---------------------------------

            hybrid_score = (
                0.70 * vector_score
                +
                0.30 * keyword_score
            )

            # ---------------------------------
            # Get chunk
            # ---------------------------------

            if chunk_key in vector_chunks:

                chunk = (
                    vector_chunks[chunk_key]
                )

            else:

                chunk = (
                    self.chunk_records[
                        chunk_key
                    ]
                )

            # ---------------------------------
            # Filter weak results
            # ---------------------------------

            if (
                hybrid_score >=
                HYBRID_THRESHOLD
            ):

                hybrid_results.append(
                    {
                        "text": chunk["text"],

                        "source":
                            chunk["source"],

                        "page":
                            chunk["page"],

                        "chunk_id":
                            chunk["chunk_id"],

                        # Vector score
                        "similarity":
                            vector_score,

                        # BM25 score
                        "keyword_score":
                            keyword_score,

                        # Final score
                        "hybrid_score":
                            hybrid_score
                    }
                )

        # =================================
        # 4. SORT BY HYBRID SCORE
        # =================================

        hybrid_results.sort(
            key=lambda result:
                result["hybrid_score"],
            reverse=True
        )

        # =================================
        # 5. RETURN TOP-K
        # =================================

        return hybrid_results[:top_k]