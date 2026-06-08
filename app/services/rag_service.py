import os
import hashlib
import json
import numpy as np
import faiss
import pickle
import uuid
import redis
import json
import fitz 
import pytesseract
from PIL import Image
import os

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
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from langchain_community.document_loaders import Docx2txtLoader



# -------------------- ENV --------------------
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
# memory_store = {}
# semantic_cache = {}
bm25 = None
bm25_corpus = []

redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)



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
        # loader = PyMuPDFLoader(str(pdf))
        documents.extend(load_document_auto(str(pdf)))

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


# -------------------- QUADRANT VECTOR STORE -----------
class QdrantVectorStore:

    def __init__(self):
        self.collection_name = "rag_collection"
        self.client = QdrantClient(host="localhost", port=6333)
        self.dim = None

    def _create_collection(self, dim):
        self.client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size= dim,   # Dimension of your embedding model
                distance=Distance.COSINE
            )
        )

    def add_embeddings(self, embeddings, metadata):
        
        points = []

        if self.dim is None:
            dim = embeddings.shape[1]
            self._create_collection(dim)

        for i, emb in enumerate(embeddings):
            points.append(PointStruct(
                id=str(uuid.uuid4()),
                vector=emb.tolist(),
                payload=metadata[i]   # text + source
            ))

        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )

    def get_all_metadata(self):
        points, _ = self.client.scroll(
        collection_name=self.collection_name,
        limit=10000
        )

        return [point.payload for point in points]
    def search(self, query_embedding, top_k=3):

        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding[0].tolist(),
            limit=top_k
        )

        return [r.payload for r in results]



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

    
    def add_embeddings(self, embeddings, metadata):
        embeddings = embeddings.astype("float32")

        # If index is not initialized
        if self.index is None:
            dim = embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dim)

        # ✅ Add new vectors to existing index
        self.index.add(embeddings)

        # ✅ Append metadata
        self.metadata.extend(metadata)


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

def rewrite_query(query: str, history):

    conversation = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history])

    prompt = f"""
    Given the conversation and the latest question, rewrite the question 
    into a clear standalone question.

    Conversation:
    {conversation}

    Question:
    {query}

    Rewritten question:
    """

    rewritten_query = generate_answer(query, prompt)

    return rewritten_query.strip()


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
vector_store = QdrantVectorStore()

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

        # ✅ Only process new/updated docs
        if docs_to_process:
            chunks = pipeline.chunk_documents(docs_to_process)
            embeddings, metadata = pipeline.embed_chunks(chunks)

            # ✅ Always upsert to Qdrant (no load/save needed)
            vector_store.add_embeddings(embeddings, metadata)

        else:
            print("[INFO] No new documents to process ✅")

        # ✅ IMPORTANT: Rebuild BM25 from ALL metadata
        # Need to fetch full metadata from Qdrant or track locally
        all_metadata = vector_store.get_all_metadata()

        bm25_corpus = [m["text"].split() for m in all_metadata]
        bm25 = BM25Okapi(bm25_corpus)

        # ✅ Save updated registry
        save_registry(registry_path, new_registry)

        INDEX_READY = True

# FAISS VECTOR STORE COMMENTED_______
# def initialize_system():
#     global INDEX_READY, bm25, bm25_corpus

#     registry_path = "faiss_store/doc_registry.json"

#     registry = load_registry(registry_path)


#     if not INDEX_READY:
#         docs_to_process = []
#         new_registry = {}

#         directory_path = Path("data").resolve()
#         pdf_files = list(directory_path.glob("**/*.pdf"))

#         for pdf in pdf_files:
#             file_hash = get_file_hash(pdf)
#             file_name = str(pdf)

#             new_registry[file_name] = file_hash

#             # ✅ Check if new or modified
#             if file_name not in registry or registry[file_name] != file_hash:
#                 print(f"[INFO] New/Updated file: {file_name}")
#                 loader = PyMuPDFLoader(str(pdf))
#                 docs_to_process.extend(loader.load())

#         if docs_to_process:            
#             chunks = pipeline.chunk_documents(docs_to_process)
#             embeddings, metadata = pipeline.embed_chunks(chunks)

            
#             if os.path.exists("faiss_store/index.faiss"):
#                 vector_store.load()
#                 vector_store.add_embeddings(embeddings, metadata)
#             else:
#                 vector_store.build(embeddings, metadata)

#             vector_store.save()

#         else:
#             print("[INFO] No new documents to process ✅")
#             vector_store.load()

#         bm25_corpus = [m["text"].split() for m in metadata]
#         bm25 = BM25Okapi(bm25_corpus)
#         save_registry(registry_path, new_registry)
#         INDEX_READY = True



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


def summarize_history(history):
    text = "\n".join(
        [f"{msg['role']}: {msg['content']}" for msg in history])

    prompt = f"""
    Summarize this conversation briefly:

    {text}
    """
    summary = generate_answer("summarize", prompt)

    return [{"role": "system", "content": summary}]


#---------------- USER QUERY FUNCTION -------------------------------
def ask_question(query: str, session_id: str):
    initialize_system()
    # history = memory_store.get(session_id, [])
    
    history_data = redis_client.get(f"chat:{session_id}")

    if history_data:
        history = json.loads(history_data)
    else:
        history = []

    
    if len(history) > 10:
        history = summarize_history(history)
    
    MAX_HISTORY = 8
    history = history[-MAX_HISTORY:]

    rewritten_query = rewrite_query(query, history)

    print(f"[REWRITTEN QUERY]: {rewritten_query}")

    query_embedding = pipeline.model.encode([rewritten_query])
    # ✅ Query Rewriting


    # for item in semantic_cache:
    #         similarity = cosine_similarity(
    #             query_embedding, item["embedding"]
    #         )[0][0]

    #         if similarity > 0.90:   # ✅ similarity threshold
    #             print("[SEMANTIC CACHE HIT ✅]")
    #             return item["answer"]
    cached = redis_client.get(query)

    if cached:
        print("[REDIS CACHE HIT ✅]")
        cached_data = json.loads(cached)
        return cached_data["answer"], cached_data["sources"]
    print("[CACHE MISS ❌]")

    # results = vector_store.search(query_embedding)
    # results = hybrid_search(query, query_embedding)
    # results = rerank_results(query, results)

    results = hybrid_search(rewritten_query, query_embedding)
    results = rerank_results(rewritten_query, results)

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

    # memory_store[session_id] = history

    redis_client.set(
    f"chat:{session_id}",
    json.dumps(history),
    ex=3600   # 1 hour expiry (important ✅)
    )
    # semantic_cache.append({
    #         "query": query,
    #         "embedding": query_embedding,
    #         "answer": answer
    #     })
    redis_client.set(
    query,
    json.dumps({
        "answer": answer,
        "sources": sources
    })
    )

    return answer, sources


# ✅ OCR for images
def extract_text_from_image(image_path):
    image = Image.open(image_path)
    return pytesseract.image_to_string(image)


# ✅ PDF with image handling
def ingest_pdf_with_images(pdf_path):
    documents = []

    # ✅ Extract text normally
    loader = PyMuPDFLoader(pdf_path)
    documents.extend(loader.load())

    # ✅ Extract images + OCR
    doc = fitz.open(pdf_path)

    for page_index in range(len(doc)):
        page = doc[page_index]
        image_list = page.get_images(full=True)

        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = doc.extract_image(xref)

            image_bytes = base_image["image"]

            image_path = f"temp_{page_index}_{img_index}.png"

            with open(image_path, "wb") as f:
                f.write(image_bytes)

            text = pytesseract.image_to_string(Image.open(image_path))

            if text.strip():
                documents.append({
                    "page_content": text,
                    "metadata": {"source": pdf_path}
                })
            os.remove(image_path)     
    return documents


# ✅ AUTO LOADER (VERY IMPORTANT)
def load_document_auto(file_path):

    ext = file_path.split(".")[-1].lower()

    if ext == "pdf":
        return ingest_pdf_with_images(file_path)

    elif ext == "txt":
        with open(file_path, "r") as f:
            return [{
                "page_content": f.read(),
                "metadata": {"source": file_path}
            }]

    elif ext == "docx":
        
        loader = Docx2txtLoader(file_path)
        return loader.load()

    elif ext in ["png", "jpg", "jpeg"]:
        text = extract_text_from_image(file_path)
        return [{
            "page_content": text,
            "metadata": {"source": file_path}
        }]

    else:
        raise ValueError("Unsupported file type")