# app/vector_store.py
import faiss
import numpy as np
import os
import pickle

class VectorStore:
    def __init__(self, dim, index_path="faiss.index", meta_path="meta.pkl"):
        self.dim = dim
        self.index_path = index_path
        self.meta_path = meta_path
        self.index = faiss.IndexFlatL2(dim)
        self.metadatas = []  # list of texts or dicts

        # optionally load existing
        if os.path.exists(self.index_path) and os.path.exists(self.meta_path):
            self.load()

    def add(self, embeddings: np.ndarray, metadatas: list):
        assert embeddings.shape[0] == len(metadatas)
        self.index.add(embeddings.astype(np.float32))
        self.metadatas.extend(metadatas)
        self.save()

    def query(self, query_emb: np.ndarray, top_k=3):
        # query_emb shape (1, dim)
        D, I = self.index.search(query_emb.astype(np.float32), top_k)
        idxs = I[0].tolist()
        results = []
        for i in idxs:
            if i < len(self.metadatas):
                results.append(self.metadatas[i])
        return results, D[0].tolist()

    def save(self):
        faiss.write_index(self.index, self.index_path)
        with open(self.meta_path, "wb") as f:
            pickle.dump(self.metadatas, f)

    def load(self):
        self.index = faiss.read_index(self.index_path)
        with open(self.meta_path, "rb") as f:
            self.metadatas = pickle.load(f)
