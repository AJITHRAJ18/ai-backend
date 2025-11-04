from sentence_transformers import SentenceTransformer
import faiss
import os

class LocalRAG:
    def __init__(self, data_dir="data"):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.index = None
        self.texts = []
        self.load_data(data_dir)
        self.build_index()

    def load_data(self, data_dir):
        for file in os.listdir(data_dir):
            with open(os.path.join(data_dir, file), "r", encoding="utf-8") as f:
                self.texts.append(f.read())

    def build_index(self):
        embeddings = self.model.encode(self.texts)
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)

    def retrieve(self, query, top_k=2):
        q_emb = self.model.encode([query])
        D, I = self.index.search(q_emb, top_k)
        return [self.texts[i] for i in I[0]]
