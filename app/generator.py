# app/generator.py
import subprocess
import json

def generate_answer(query: str, docs: list, max_tokens=250):
    context = "\n\n---\n\n".join(docs)
    prompt = (
        f"You are a helpful assistant. Use the following context to answer briefly.\n\n"
        f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    )

    result = subprocess.run(
        ["ollama", "run", "llama3"],
        input=prompt.encode("utf-8"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    text = result.stdout.decode("utf-8").strip()
    return text
