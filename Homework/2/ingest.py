import os
import hashlib
import chromadb
import ollama
import nbformat
from PIL import Image
import pytesseract
import markdown as md

REPO = "data/Prompt-Engineering-Guide"
DB_DIR = "chroma_db"
COLLECTION = "peg"

def hash_id(text):
    return hashlib.sha1(text.encode()).hexdigest()

def read_file(path):
    ext = os.path.splitext(path)[1].lower()

    if ext in [".md", ".mdx"]:
        return md.markdown(open(path, encoding="utf-8").read())

    if ext == ".ipynb":
        nb = nbformat.read(path, as_version=4)
        return "\n".join(c.source for c in nb.cells if c.cell_type == "markdown")

    if ext == ".png":
        return pytesseract.image_to_string(Image.open(path))

    return ""

def summarize(text):
    prompt = f"Summarize this for prompt engineering knowledge:: {text[:4000]}"
    r = ollama.generate(
        model="llama3.2:latest",
        prompt=prompt
    )
    print("\nSummary:\n", r["response"])
    return r["response"]

client = chromadb.PersistentClient()
col = client.get_or_create_collection(COLLECTION)

for root, _, files in os.walk(REPO):
    for f in files:
        if f.endswith((".md", ".mdx", ".ipynb", ".png")):
            path = os.path.join(root, f)
            print('Path', path)
            text = read_file(path)

            summary = summarize(text)
            emb = ollama.embeddings(
                model="llama3.2:latest",
                prompt=summary
            )["embedding"]

            col.add(
                ids=[hash_id(path)],
                documents=[summary],
                embeddings=[emb],
                metadatas=[{"source": path}]
            )

print("DONE")
