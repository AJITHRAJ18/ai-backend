# app/main.py
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from app.retriever import Retriever
from app.generator import generate_answer

app = FastAPI()
retriever = Retriever()  # loads/indexes local docs

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"status": "alive"}

@app.get("/search")
def search(query: str = Query(..., min_length=1)):
    try:
        docs, distances = retriever.retrieve(query, top_k=3)
        answer = generate_answer(query, docs)
        return {
            "query": query,
            "retrieved_docs": docs,
            "distances": distances,
            "final_answer": answer
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
