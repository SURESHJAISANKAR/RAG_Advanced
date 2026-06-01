import os
from pathlib import Path
from typing import List, Any, Dict
import numpy as np
import faiss
import pickle

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

from groq import Groq
from dotenv import load_dotenv

# -------------------- ENV --------------------
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# -------------------- DATA LOADER --------------------
def load_all_docs(data_dir: str) -> List[Any]:
    directory_path = Path(data_dir).resolve()
    documents = []

    pdf_files = list(directory_path.glob("**/*.pdf"))
    print(f"[INFO] Loaded {len(pdf_files)} PDFs")

    for pdf in pdf_files:
        loader = PyMuPDFLoader(str(pdf))
        documents.extend(loader.load())

    return documents


# -------------------- EMBEDDING --------------------
class EmbeddingPipeline:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def chunk_documents(self, documents: List[Any]):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )
        return splitter.split_documents(documents)

    def embed_chunks(self, chunks):
        texts = [chunk.page_content for chunk in chunks]
        embeddings = self.model.encode(texts)

        metadata = [
            {"text": chunk.page_content, "source": chunk.metadata.get("source")}
            for chunk in chunks
        ]

        return embeddings, metadata


# -------------------- VECTOR STORE --------------------
class FaissVectorStore:
    def __init__(self):
        self.path = "faiss_store"
        os.makedirs(self.path, exist_ok=True)
        self.index = None
        self.metadata = []

    def build(self, embeddings, metadata):
        dim = embeddings.shape[1]

        self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings.astype("float32"))
        self.metadata = metadata

        faiss.write_index(self.index, f"{self.path}/index.faiss")

        with open(f"{self.path}/metadata.pkl", "wb") as f:
            pickle.dump(self.metadata, f)

    def load(self):
        self.index = faiss.read_index(f"{self.path}/index.faiss")

        with open(f"{self.path}/metadata.pkl", "rb") as f:
            self.metadata = pickle.load(f)

    def search(self, query_embedding, top_k=3):
        D, I = self.index.search(query_embedding.astype("float32"), top_k)

        results = []
        for idx in I[0]:
            results.append(self.metadata[idx])

        return results


# -------------------- LLM --------------------
def generate_answer(question: str, context: str):
    prompt = f"""
    Answer ONLY from context.
    If not found say "I don't know".

    Context:
    {context}

    Question:
    {question}
    """

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content


# -------------------- MAIN FUNCTION --------------------
pipeline = EmbeddingPipeline()
vector_store = FaissVectorStore()

INDEX_READY = False

def initialize_system():
    global INDEX_READY

    if not INDEX_READY:
        if os.path.exists("faiss_store/index.faiss"):
            print("[INFO] Loading existing index...")
            vector_store.load()
        else:
            print("[INFO] Building index first time...")
            docs = load_all_docs("data")
            chunks = pipeline.chunk_documents(docs)
            embeddings, metadata = pipeline.embed_chunks(chunks)
            vector_store.build(embeddings, metadata)

        INDEX_READY = True


def ask_question(query: str):
    initialize_system()

    query_embedding = pipeline.model.encode([query])
    results = vector_store.search(query_embedding)

    context = "\n".join([r["text"] for r in results])

    answer = generate_answer(query, context)

    return answer