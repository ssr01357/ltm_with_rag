import os
import json
import pickle
import torch
import torch.nn.functional as F

from torch.nn import DataParallel
from typing import List, Dict, Any

from transformers import AutoTokenizer, AutoModel
from langchain_community.retrievers import BM25Retriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv, find_dotenv
from tqdm import tqdm

# ------------------
# Environment Setup
# ------------------
def setup_environment():
    """
    Loads environment variables and configures CUDA usage.
    """
    env_path = find_dotenv()
    if not env_path:
        # Fallback path - adjust to your environment as needed
        env_path = "/home/yl3427/.env"

    if not load_dotenv(env_path):
        raise Exception("Failed to load .env file")
    
    # Set GPU settings as needed
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"


# --------------------
# Data Loading
# --------------------
def load_cases(json_path: str) -> Dict[str, Any]:
    """
    Loads case data from a JSON file.
    Args:
        json_path: path to the JSON file containing {hadm_id: {...}} mappings.
    Returns:
        Dictionary of case data.
    """
    with open(json_path, "r") as f:
        cases = json.load(f)
    return cases


# --------------------
# Text Splitting
# --------------------
def create_documents(cases: Dict[str, Any],
                     tokenizer,
                     max_length: int = 512) -> List[Document]:
    """
    Splits clinical text into smaller chunks.

    Args:
        cases: dictionary of hadm_id -> {before_diagnosis, after_diagnosis}
        tokenizer: HF tokenizer
        max_length: maximum token chunk size
    Returns:
        List of Document objects suitable for further embedding.
    """
    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        separators=[
            "\n\n", "\n", r'(?<=[.?"\s])\s+', " ", ".", ","
        ],
        tokenizer=tokenizer,
        chunk_size=max_length,
        chunk_overlap=20,
        add_start_index=True,
        strip_whitespace=True,
        is_separator_regex=True
    )

    all_docs = []
    unique_texts = set()

    for hadm_id, data in cases.items():
        full_text = data["before_diagnosis"]
        docs = text_splitter.create_documents(
            texts=[full_text],
            metadatas=[{
                "hadm_id": hadm_id,
                "full_text": full_text,
                "diagnosis": data["after_diagnosis"]
            }]
        )
        # Deduplicate
        for d in docs:
            if d.page_content not in unique_texts:
                unique_texts.add(d.page_content)
                all_docs.append(d)

    return all_docs


# --------------------
# Embedding & Chroma
# --------------------
def embed_docs_in_chroma(docs: List[Document],
                         embedding_model,
                         collection,
                         max_length: int = 512):
    """
    Embed documents and store them in a Chroma collection.

    Args:
        docs: List of Document objects to embed.
        embedding_model: The embedding model with .encode() method.
        collection: The Chroma collection to store embeddings.
        max_length: maximum token length for the embedding model.
    """
    pbar = tqdm(total=len(docs), desc="Embedding Documents")
    for doc in docs:
        doc_text = doc.page_content
        doc_meta = doc.metadata
        doc_id = f"{doc_meta['hadm_id']}_{doc_meta['start_index']}"

        with torch.no_grad():
            embeddings = embedding_model.encode(
                [doc_text],
                instruction="",
                max_length=max_length
            )
            embeddings = embeddings.cpu().numpy().tolist()

        collection.add(
            embeddings=embeddings,
            documents=[doc_text],
            metadatas=[doc_meta],
            ids=[doc_id],
        )
        pbar.update(1)
        torch.cuda.empty_cache()
    pbar.close()


# --------------------
# Query Function
# --------------------
def hybrid_query(cases: Dict[str, Any],
                 docs: List[Document],
                 collection,
                 embedding_model,
                 query_text: str,
                 query_prefix: str,
                 max_length: int = 512,
                 semantic_k: int = 5,
                 bm25_k: int = 5,
                 bm25_weight: float = 0.5) -> List[str]:
    """
    Performs a hybrid query using semantic (Chroma) + BM25.

    Args:
        cases: Original dictionary of hadm_id -> {before_diagnosis, after_diagnosis}
        docs: The Document objects used for BM25 indexing.
        collection: Chroma collection to query.
        embedding_model: The embedding model with .encode() method.
        query_text: The raw query text (clinical note).
        query_prefix: Additional instruction prefix for the embedding model.
        max_length: The token limit for the embedding model.
        bm25_k: The top-k BM25 documents to consider.
        bm25_weight: Weight assigned to BM25 scores vs. semantic scores (1 - bm25_weight).
    Returns:
        A list of retrieved document strings (before_diagnosis + discharge diagnosis).
    """
    # --- Semantic Retrieval ---
    query_embedding = embedding_model.encode(
        [query_text],
        instruction=query_prefix,
        max_length=max_length
    ).cpu().numpy().tolist()

    results_semantic = collection.query(
        query_embeddings=query_embedding,
        n_results=semantic_k,  # Top-k from Chroma
    )
    # e.g. ["hadmID_chunkindex", ...]
    results_semantic_ids = [full_id.split("_")[0] for full_id in results_semantic["ids"][0]]

    # --- BM25 Retrieval ---
    bm25_retriever = BM25Retriever.from_documents(docs, k=bm25_k)
    results_bm25 = bm25_retriever.get_relevant_documents(query_text)
    results_bm25_ids = [doc.metadata["hadm_id"] for doc in results_bm25]

    # Union IDs from both retrievals
    combined_ids = set(results_semantic_ids) | set(results_bm25_ids)

    # Weighted scoring
    semantic_weight = 1.0 - bm25_weight
    ids_to_score = {}

    for chunk_id in combined_ids:
        score = 0.0
        if chunk_id in results_semantic_ids:
            idx_sem = results_semantic_ids.index(chunk_id)
            score += semantic_weight * (1 / (idx_sem + 1))
        if chunk_id in results_bm25_ids:
            idx_bm25 = results_bm25_ids.index(chunk_id)
            score += bm25_weight * (1 / (idx_bm25 + 1))
        ids_to_score[chunk_id] = score

    # Sort by combined score
    sorted_ids = sorted(ids_to_score.keys(), key=lambda x: ids_to_score[x], reverse=True)

    # Return combined text
    retrieved_docs = []
    for doc_id in sorted_ids:
        before = cases[doc_id]["before_diagnosis"]
        after = cases[doc_id]["after_diagnosis"]
        retrieved_docs.append(f"{before}\nDischarge Diagnosis: {after}")

    return retrieved_docs


def main():
    """
    Main function to run the RAG pipeline:
      1. Setup environment
      2. Load cases
      3. Create documents (split text)
      4. Connect to Chroma, create/get collection
      5. Embed into Chroma
      6. Example query for testing
    """
    # 1) Environment Setup
    setup_environment()

    # 2) Load Cases
    json_path = "/secure/shared_data/SOAP/MIMIC/cases_base.json"
    cases = load_cases(json_path)

    # 3) Create Documents
    dir_path = "/secure/shared_data/rag_embedding_model"
    tokenizer = AutoTokenizer.from_pretrained("nvidia/NV-Embed-v2",
                                              trust_remote_code=True,
                                              cache_dir=dir_path)
    docs_processed = create_documents(cases, tokenizer, max_length=512)

    # 4) Connect to Chroma
    client = chromadb.PersistentClient(
        path="/secure/shared_data/rag_embedding_model/chroma_db",
        settings=Settings(allow_reset=True)
    )
    mimic_collection = client.get_or_create_collection(
        name="mimic_3prob",
        metadata={"hnsw:space": "cosine"}
    )

    # 5) Load/Initialize Embedding Model & Embed Docs
    embedding_model = AutoModel.from_pretrained("nvidia/NV-Embed-v2",
                                                trust_remote_code=True,
                                                cache_dir=dir_path,
                                                device_map="auto")
    # Optional: If you haven't already embedded docs, uncomment to embed
    # embed_docs_in_chroma(docs_processed, embedding_model, mimic_collection, max_length=512)

    # 6) Run Example Query
    query_prefix = (
        "Given the following clinical note, retrieve the most similar "
        "clinical case. The clinical note is:\n\n"
    )
    query_text = (
        "pleuritic right chest pain\n- patient started on coumadin\n..."
        "respiratory support\no2 delivery device: nasal cannula\nspo2: 98%\n..."
        # Truncated for brevity...
    )

    retrieved = hybrid_query(
        cases,
        docs_processed,
        mimic_collection,
        embedding_model,
        query_text=query_text,
        query_prefix=query_prefix,
        max_length=512,
        semantic_k=5,
        bm25_k=5,
        bm25_weight=0.5
    )

    # Print top retrieved docs (for example)
    for idx, doc in enumerate(retrieved, start=1):
        print(f"\n--- Retrieved Doc #{idx} ---")
        print(doc)


if __name__ == "__main__":
    main()
