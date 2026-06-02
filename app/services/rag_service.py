import os
import hashlib
import json
import numpy as np
import faiss
import pickle
from pathlib import Path
from typing import List, Any, Dict
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from sentence_transformers import CrossEncoder
from groq import Groq
from dotenv import load_dotenv


# -------------------- ENV --------------------
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
memory_store = {}
semantic_cache = {}
bm25 = None
bm25_corpus = []

#Hashing for Docs
def get_file_hash(file_path):
    with open(file_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def load_registry(path):
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}

def save_registry(path, data):
    with open(path, "w") as f:
        json.dump(data, f)

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
def generate_answer(question: str, full_prompt: str):
    # prompt = f"""
    # Answer ONLY from context.
    # If not found say "I don't know".

    # Context:
    # {context}

    # Question:
    # {question}
    # """

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": full_prompt}]
    )

    return response.choices[0].message.content


# -------------------- MAIN FUNCTION --------------------
pipeline = EmbeddingPipeline()
vector_store = FaissVectorStore()

INDEX_READY = False

def initialize_system():
    global INDEX_READY, bm25, bm25_corpus

    registry_path = "faiss_store/doc_registry.json"

    registry = load_registry(registry_path)


    if not INDEX_READY:
        docs_to_process = []
        new_registry = {}

        directory_path = Path("data").resolve()
        pdf_files = list(directory_path.glob("**/*.pdf"))

        for pdf in pdf_files:
            file_hash = get_file_hash(pdf)
            file_name = str(pdf)

            new_registry[file_name] = file_hash

            # ✅ Check if new or modified
            if file_name not in registry or registry[file_name] != file_hash:
                print(f"[INFO] New/Updated file: {file_name}")
                loader = PyMuPDFLoader(str(pdf))
                docs_to_process.extend(loader.load())

        if docs_to_process:            
            chunks = pipeline.chunk_documents(docs_to_process)
            embeddings, metadata = pipeline.embed_chunks(chunks)

            
            if os.path.exists("faiss_store/index.faiss"):
                vector_store.load()
                vector_store.add_embeddings(embeddings, metadata)
            else:
                vector_store.build(embeddings, metadata)

            vector_store.save()

        else:
            print("[INFO] No new documents to process ✅")
            vector_store.load()

        bm25_corpus = [m["text"].split() for m in metadata]
        bm25 = BM25Okapi(bm25_corpus)
        save_registry(registry_path, new_registry)
        INDEX_READY = True

def rerank_results(query: str, results: list, top_k=3):

    # ✅ Prepare (query, doc) pairs
    pairs = [(query, r["text"]) for r in results]

    # ✅ Get scores
    scores = reranker.predict(pairs)

    # ✅ Combine results with scores
    scored_results = list(zip(results, scores))

    # ✅ Sort descending by score
    scored_results.sort(key=lambda x: x[1], reverse=True)

    # ✅ Take top_k results
    top_results = [item[0] for item in scored_results[:top_k]]

    return top_results

def hybrid_search(query: str, query_embedding):

    # ✅ Vector search
    vector_results = vector_store.search(query_embedding)

    # ✅ BM25 search
    tokenized_query = query.split()
    scores = bm25.get_scores(tokenized_query)

    top_n = np.argsort(scores)[-3:]  # top 3 BM25 results

    bm25_results = [vector_store.metadata[i] for i in top_n]

    # ✅ Combine results
    combined_results = vector_results + bm25_results

    return combined_results


def ask_question(query: str, session_id: str):
    initialize_system()
    history = memory_store.get(session_id, [])

    query_embedding = pipeline.model.encode([query])

    for item in semantic_cache:
            similarity = cosine_similarity(
                query_embedding, item["embedding"]
            )[0][0]

            if similarity > 0.90:   # ✅ similarity threshold
                print("[SEMANTIC CACHE HIT ✅]")
                return item["answer"]

    print("[CACHE MISS ❌]")

    # results = vector_store.search(query_embedding)
    results = hybrid_search(query, query_embedding)

    results = rerank_results(query, results)

    context = "\n".join([r["text"] for r in results])

    
    conversation = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history])

    
    full_prompt = f"""
        Use the conversation history and context to answer the question.

        Conversation History:
        {conversation}

        Context:
        {context}

        Question:
        {query}
        """

    answer = generate_answer(query, full_prompt)

    sources = list(
        set([r["source"] for r in results if r.get("source")])
    )

    history.append({"role": "user", "content": query})
    history.append({"role": "assistant", "content": answer})

    memory_store[session_id] = history
    semantic_cache.append({
            "query": query,
            "embedding": query_embedding,
            "answer": answer
        })

    return answer, sources