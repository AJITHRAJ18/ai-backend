# app/embedder.py
from sentence_transformers import SentenceTransformer
import numpy as np

class Embedder:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_texts(self, texts):
        # returns numpy array shape (n, dim)
        embs = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return embs

    def embed_query(self, query):
        emb = self.model.encode([query], convert_to_numpy=True, show_progress_bar=False)
        return emb
