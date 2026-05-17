"""
Hybrid Retriever: FAISS (dense) + BM25 (sparse) with Reciprocal Rank Fusion.
"""

import json
import os
import numpy as np
from rank_bm25 import BM25Okapi

# Try to use faiss-cpu
try:
    import faiss
except ImportError:
    faiss = None


class HybridRetriever:
    """Combines FAISS semantic search with BM25 keyword search using RRF."""

    def __init__(self, catalog_path: str = None):
        if catalog_path is None:
            catalog_path = os.path.join(os.path.dirname(__file__), "data", "catalog.json")

        with open(catalog_path, "r", encoding="utf-8") as f:
            self.catalog = json.load(f)

        # Build a URL lookup for validation
        self.valid_urls = {item["url"] for item in self.catalog}

        # Create rich text representations for each item
        self.documents = []
        self.tokenized_docs = []
        for item in self.catalog:
            doc = self._build_document_text(item)
            self.documents.append(doc)
            self.tokenized_docs.append(doc.lower().split())

        # Initialize BM25
        print(f"Building BM25 index over {len(self.catalog)} items...")
        self.bm25 = BM25Okapi(self.tokenized_docs)

        # Initialize FAISS (optional) with precomputed embeddings when available.
        # Disable by default to fit low-memory deployments; set ENABLE_DENSE_RETRIEVAL=true to enable.
        index_path = os.path.join(os.path.dirname(__file__), "data", "catalog.faiss")
        embeddings_path = os.path.join(os.path.dirname(__file__), "data", "embeddings.npy")

        self.embedder = None

        self.index = None
        self.embeddings = None
        dense_enabled = os.getenv("ENABLE_DENSE_RETRIEVAL", "false").strip().lower() in {
            "1",
            "true",
            "yes",
        }

        if not dense_enabled:
            print("Dense retrieval disabled (ENABLE_DENSE_RETRIEVAL=false). Using BM25-only.")
        elif faiss is None:
            print("WARNING: faiss not available. Falling back to BM25-only retrieval.")
        elif os.path.exists(index_path) and os.path.exists(embeddings_path):
            print("Loading pre-built FAISS index...")
            self.index = faiss.read_index(index_path)
            self.embeddings = np.load(embeddings_path)
        else:
            print("FAISS embeddings not found. Falling back to BM25-only retrieval.")

        print(f"Retriever ready: {len(self.catalog)} items indexed.")

    def _build_document_text(self, item: dict) -> str:
        """Create a rich text representation of a catalog item for embedding."""
        parts = [item["name"]]

        if item.get("description"):
            parts.append(item["description"])

        if item.get("test_type_labels"):
            parts.append("Test types: " + ", ".join(item["test_type_labels"]))

        if item.get("test_type"):
            parts.append("Codes: " + ", ".join(item["test_type"]))

        if item.get("duration"):
            parts.append(f"Duration: {item['duration']}")

        if item.get("languages_raw"):
            parts.append(f"Languages: {item['languages_raw']}")

        if item.get("remote_testing"):
            parts.append("Remote testing available")

        if item.get("adaptive_irt"):
            parts.append("Adaptive/IRT")

        return " | ".join(parts)

    def search_bm25(self, query: str, top_k: int = 15) -> list[tuple[int, float]]:
        """BM25 keyword search. Returns list of (index, score)."""
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(int(idx), float(scores[idx])) for idx in top_indices if scores[idx] > 0]

    def search_faiss(self, query: str, top_k: int = 15) -> list[tuple[int, float]]:
        """FAISS semantic search. Returns list of (index, score)."""
        if self.index is None:
            return []
        if self.embedder is None:
            # Lazy-load only when dense retrieval is actually possible
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                print("WARNING: sentence-transformers not installed. Dense retrieval disabled.")
                return []
            self.embedder = SentenceTransformer("all-MiniLM-L6-v2")

        query_embedding = self.embedder.encode([query], normalize_embeddings=True).astype("float32")
        scores, indices = self.index.search(query_embedding, top_k)
        return [(int(idx), float(score)) for idx, score in zip(indices[0], scores[0]) if idx >= 0]

    def reciprocal_rank_fusion(
        self,
        bm25_results: list[tuple[int, float]],
        faiss_results: list[tuple[int, float]],
        k: int = 60,
    ) -> list[int]:
        """Combine BM25 and FAISS results using Reciprocal Rank Fusion."""
        rrf_scores = {}

        for rank, (idx, _) in enumerate(bm25_results):
            rrf_scores[idx] = rrf_scores.get(idx, 0) + 1.0 / (k + rank + 1)

        for rank, (idx, _) in enumerate(faiss_results):
            rrf_scores[idx] = rrf_scores.get(idx, 0) + 1.0 / (k + rank + 1)

        # Sort by RRF score descending
        sorted_indices = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        return sorted_indices

    def retrieve(self, query: str, top_k: int = 10) -> list[dict]:
        """
        Hybrid retrieval: BM25 + FAISS + RRF.
        Returns top_k catalog items with their metadata.
        """
        bm25_results = self.search_bm25(query, top_k=15)
        faiss_results = self.search_faiss(query, top_k=15)
        fused_indices = self.reciprocal_rank_fusion(bm25_results, faiss_results)

        results = []
        for idx in fused_indices[:top_k]:
            item = self.catalog[idx].copy()
            item["_document_text"] = self.documents[idx]
            results.append(item)

        return results

    def validate_url(self, url: str) -> bool:
        """Check if a URL exists in the catalog."""
        return url in self.valid_urls

    def find_by_url(self, url: str) -> dict | None:
        """Find a catalog item by URL."""
        for item in self.catalog:
            if item["url"] == url:
                return item
        return None

    def find_by_name(self, name: str) -> dict | None:
        """Find a catalog item by exact or partial name match."""
        name_lower = name.lower().strip()
        # Try exact match first
        for item in self.catalog:
            if item["name"].lower().strip() == name_lower:
                return item
        # Try partial match
        for item in self.catalog:
            if name_lower in item["name"].lower():
                return item
        return None


if __name__ == "__main__":
    # Quick test
    retriever = HybridRetriever()
    
    test_queries = [
        "Java developer assessment",
        "OPQ32r",
        "contact center customer service",
        "safety plant operator",
        "Excel Word admin assistant",
    ]
    
    for query in test_queries:
        print(f"\n--- Query: '{query}' ---")
        results = retriever.retrieve(query, top_k=5)
        for i, item in enumerate(results):
            print(f"  {i+1}. {item['name']} [{','.join(item.get('test_type', []))}]")
