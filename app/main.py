from fastapi import FastAPI
from app.rag_engine import LocalRAG

app = FastAPI()
rag = LocalRAG()

@app.get("/health")
def health():
    return {"status": "alive"}

@app.get("/search")
def search(query: str):
    results = rag.retrieve(query)
    return {"query": query, "results": results}
