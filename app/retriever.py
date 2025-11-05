# app/retriever.py
import os

from .embedder import Embedder
from .vector_store import VectorStore

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

class Retriever:
    def __init__(self):
        self.embedder = Embedder()
        # initialize using embedding dim
        sample_emb = self.embedder.embed_query("sample")
        dim = sample_emb.shape[1]
        self.store = VectorStore(dim=dim)
        if not self.store.metadatas:
            self._index_local_files(DATA_DIR)

    def _index_local_files(self, data_dir):
        texts = []
        for fname in sorted(os.listdir(data_dir)):
            if fname.lower().endswith(".txt"):
                path = os.path.join(data_dir, fname)
                with open(path, "r", encoding="utf-8") as f:
                    texts.append(f.read())
        if not texts:
            return
        embs = self.embedder.embed_texts(texts)
        metadatas = texts
        self.store.add(embs, metadatas)

    def retrieve(self, query, top_k=3):
        q_emb = self.embedder.embed_query(query)
        docs, distances = self.store.query(q_emb, top_k=top_k)
        return docs, distances
