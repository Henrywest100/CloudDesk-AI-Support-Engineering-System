"""
CloudDesk AI Support Engineer — Clean Retrieval & Inference Pipeline
====================================================================
Connects directly to the pre-indexed Vector Database (Pinecone / ChromaDB),
retrieves relevant context, formats citations, and generates grounded answers
using HuggingFace InferenceClient.
"""

import os
import time
import pandas as pd
from typing import Tuple, List, Dict, Any
from dotenv import load_dotenv

from huggingface_hub import InferenceClient
from sentence_transformers import SentenceTransformer
import chromadb

try:
    from pinecone import Pinecone
except ImportError:
    Pinecone = None

# -------------------------------------------------------------
# 1. System Configuration & Prompts
# -------------------------------------------------------------
SYSTEM_PROMPT = """You are the CloudDesk AI Support Engineer.
Answer customer support questions accurately based ONLY on the provided context.
Provide a clear step-by-step resolution and cite source documents.
If the answer is not present in the context or confidence is low, state that clearly and suggest contacting Tier-2 Support."""

FALLBACK_MESSAGE = (
    "[ESCALATED] Answer Confidence Below Threshold (< 60%) — Auto-Escalated to Human Support\n\n"
    "This query has been escalated to CloudDesk Support Engineering for assistance."
)

def load_config() -> Dict[str, Any]:
    load_dotenv()
    return {
        "hf_token": os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_API_KEY"),
        "hf_model": os.getenv("HF_MODEL", "deepseek-ai/DeepSeek-V4-Pro:novita"),
        "pinecone_api_key": os.getenv("PINECONE_API_KEY"),
        "pinecone_index": os.getenv("PINECONE_INDEX_NAME", "clouddesk-support-rag"),
        "embed_model": os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
        "vector_db_dir": os.getenv("VECTOR_DB_DIR", "vector_db"),
        "confidence_threshold": 0.60
    }

# -------------------------------------------------------------
# 2. Vector Retriever (Pinecone & Chroma)
# -------------------------------------------------------------
def get_vector_store(cfg: Dict[str, Any]):
    embedder = SentenceTransformer(cfg["embed_model"])
    
    # Connect to Pinecone index if API key is provided
    if cfg.get("pinecone_api_key") and Pinecone:
        try:
            pc = Pinecone(api_key=cfg["pinecone_api_key"])
            pc_index = pc.Index(cfg["pinecone_index"])
            return ("pinecone", pc_index, embedder)
        except Exception as e:
            print(f"[*] Pinecone connection notice: {e}. Using local Chroma DB.")

    # Local Chroma store connection
   # client = chromadb.PersistentClient(path=cfg["vector_db_dir"])
    client = chromadb.PersistentClient(path=os.path.join(cfg["vector_db_dir"], "chroma"))
   
    coll = client.get_or_create_collection("clouddesk_support_kb")
    return ("chroma", coll, embedder)

def retrieve_docs(vstore, query: str, k: int = 3) -> List[Dict[str, Any]]:
    store_type, index_or_coll, embedder = vstore
    docs = []
    
    if store_type == "pinecone":
        q_emb = embedder.encode([query])[0].tolist()
        res = index_or_coll.query(vector=q_emb, top_k=k, include_metadata=True)
        for m in res.get("matches", []):
            docs.append({
                "text": m.get("metadata", {}).get("text", "") or m.get("metadata", {}).get("page_content", ""),
                "title": m.get("metadata", {}).get("title", "CloudDesk Doc"),
                "source_type": m.get("metadata", {}).get("source_type", "doc"),
                "score": float(m.get("score", 0.5))
            })
    else:
        q_emb = embedder.encode([query]).tolist()
        res = index_or_coll.query(query_embeddings=q_emb, n_results=k)
        if res and res.get("documents"):
            for text, meta, dist in zip(res["documents"][0], res["metadatas"][0], res.get("distances", [[0.5]*k])[0]):
                score = max(0.0, 1.0 - float(dist)) if float(dist) <= 1.0 else max(0.0, 1.0 / (1.0 + float(dist)))
                docs.append({
                    "text": text,
                    "title": meta.get("title", "CloudDesk Doc"),
                    "source_type": meta.get("source_type", "doc"),
                    "score": round(score, 4)
                })
    return docs

# -------------------------------------------------------------
# 3. Document Formatter & Confidence Scoring
# -------------------------------------------------------------
def format_docs(docs: List[Dict[str, Any]]) -> Tuple[str, str]:
    blocks, cites = [], []
    for d in docs:
        cites.append(f"- **[{d['title']}]** (`{d['source_type']}`) — Similarity: {int(d['score']*100)}%")
        blocks.append(f"[{d['title']} ({d['source_type']})]\n{d['text']}")
    return "\n\n---\n\n".join(blocks), "\n".join(cites)

def compute_confidence(docs: List[Dict[str, Any]]) -> float:
    if not docs:
        return 0.0
    top_score = docs[0]["score"]
    avg_score = sum(d["score"] for d in docs) / float(len(docs))
    return round(0.6 * top_score + 0.4 * avg_score, 2)

# -------------------------------------------------------------
# 4. HuggingFace Chat Completion & RAG Execution
# -------------------------------------------------------------
def call_hf_chat(client: InferenceClient, model_id: str, system: str, user_content: str) -> str:
    completion = client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ],
        temperature=0.2,
        max_tokens=512,
    )
    return completion.choices[0].message.content

def run_rag_pipeline(cfg: Dict[str, Any], vstore, client: InferenceClient, query: str) -> Dict[str, Any]:
    docs = retrieve_docs(vstore, query, k=3)
    context, cites = format_docs(docs)
    confidence = compute_confidence(docs)
    requires_escalation = confidence < cfg["confidence_threshold"]

    # ← NEW: check escalation FIRST, before calling LLM
    if requires_escalation:
        return {
            "query": query,
            "answer": FALLBACK_MESSAGE,
            "confidence": confidence,
            "confidence_pct": f"{int(confidence*100)}%",
            "requires_escalation": True,
            "citations": cites,
            "docs": docs
        }

    # Only call LLM if confidence is above threshold
    user_prompt = f"Customer Question: {query}\n\nContext:\n{context}\n\nAnswer:"
    answer = ""
    if client and cfg.get("hf_token"):
        try:
            answer = call_hf_chat(client, cfg["hf_model"], SYSTEM_PROMPT, user_prompt)
        except Exception as e:
            print(f"[*] HuggingFace API notice: {e}")

    # Fallback if LLM failed but confidence is OK
    if not answer:
        top_doc = docs[0] if docs else {"title": "Doc", "text": "No info available."}
        clean_text = top_doc["text"].strip().replace("\n", "\n> ")
        answer = (
            f"### Resolution Guide for CloudDesk Customer Support\n"
            f"**Confidence:** {int(confidence*100)}% | **Source:** [{top_doc['title']}]\n\n"
            f"Based on CloudDesk documentation:\n"
            f"> {clean_text[:500]}\n\n"
            f"**Verified Sources:**\n{cites}"
        )

    return {
        "query": query,
        "answer": answer,
        "confidence": confidence,
        "confidence_pct": f"{int(confidence*100)}%",
        "requires_escalation": False,
        "citations": cites,
        "docs": docs
    }
# -------------------------------------------------------------
# 5. Evaluation Benchmark Suite
# -------------------------------------------------------------
def evaluate_benchmark(cfg: Dict[str, Any], vstore, client: InferenceClient, csv_path: str = None) -> Dict[str, Any]:
    if csv_path is None:
        csv_path = os.path.join("clouddesk_dataset", "support_tickets", "previous_support_tickets.csv")

    if not os.path.exists(csv_path):
        print(f"[!] Test CSV path '{csv_path}' not found.")
        return {}

    df = pd.read_csv(csv_path)
    print(f"[*] Running RAG Evaluation Benchmark across {len(df)} historical tickets...")

    hits_1, hits_3, hits_5 = 0, 0, 0
    confidences = []
    escalations = 0
    records = []

    for idx, r in df.iterrows():
        tkt_id = str(r.get("ticket_id", f"tkt_{idx}"))
        query = str(r.get("customer_question", ""))
        ref_title = str(r.get("source_title_reference", "")).strip().lower()

        res = run_rag_pipeline(cfg, vstore, client, query)
        conf = res["confidence"]
        confidences.append(conf)

        if res["requires_escalation"]:
            escalations += 1

        hit_1, hit_3, hit_5 = False, False, False
        retrieved_docs = res["docs"]

        for rank, doc in enumerate(retrieved_docs, 1):
            d_title = doc.get("title", "").lower()
            is_match = (ref_title in d_title or d_title in ref_title) if ref_title else True
            if is_match:
                if rank == 1: hit_1 = True
                if rank <= 3: hit_3 = True
                if rank <= 5: hit_5 = True
                break

        if hit_1: hits_1 += 1
        if hit_3: hits_3 += 1
        if hit_5: hits_5 += 1

        records.append({
            "ticket_id": tkt_id,
            "question": query,
            "ref_title": r.get("source_title_reference", ""),
            "top_retrieved": retrieved_docs[0]["title"] if retrieved_docs else "None",
            "confidence": conf,
            "escalated": res["requires_escalation"],
            "hit_at_1": hit_1,
            "hit_at_3": hit_3
        })

    total = float(len(df))
    metrics = {
        "total_tickets": len(df),
        "hit_rate_at_1_pct": f"{round((hits_1 / total) * 100, 2)}%",
        "hit_rate_at_3_pct": f"{round((hits_3 / total) * 100, 2)}%",
        "hit_rate_at_5_pct": f"{round((hits_5 / total) * 100, 2)}%",
        "mean_confidence": round(float(pd.Series(confidences).mean()), 4),
        "total_escalations": escalations,
        "escalation_rate_pct": f"{round((escalations / total) * 100, 2)}%",
        "records": records
    }

    print("\n========================================================")
    print("          CLOUDDESK RAG BENCHMARK EVALUATION RESULTS     ")
    print("========================================================")
    print(f" Total Tickets Tested : {metrics['total_tickets']}")
    print(f" Hit Rate @ 1         : {metrics['hit_rate_at_1_pct']}")
    print(f" Hit Rate @ 3         : {metrics['hit_rate_at_3_pct']}")
    print(f" Hit Rate @ 5         : {metrics['hit_rate_at_5_pct']}")
    print(f" Mean Confidence      : {metrics['mean_confidence']}")
    print(f" Escalations (<60%)   : {metrics['total_escalations']} ({metrics['escalation_rate_pct']})")
    print("========================================================\n")

    return metrics

# -------------------------------------------------------------
# Quick Execution Entry Point
# -------------------------------------------------------------
if __name__ == "__main__":
    cfg = load_config()
    print("[*] Connecting to Vector Store...")
    vstore = get_vector_store(cfg)

    client = InferenceClient(api_key=cfg["hf_token"]) if cfg.get("hf_token") else None

    sample_query = "My SAML login stopped working after adding a new domain"
    print(f"\n[*] Testing Sample Query: '{sample_query}'\n")
    res = run_rag_pipeline(cfg, vstore, client, sample_query)

    print(f"Confidence : {res['confidence_pct']}")
    print(f"Escalated  : {res['requires_escalation']}")
    print(f"\nAnswer:\n{res['answer']}")
