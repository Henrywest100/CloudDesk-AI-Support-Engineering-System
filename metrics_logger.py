"""Simple metrics logger for CloudDesk."""
import json
import os
from datetime import datetime

METRICS_FILE = "metrics_log.jsonl"

def log_query(question: str, confidence: float, escalated: bool,
              sources_used: int, vector_store: str, latency_s: float):
    """Append one query record to the log file."""
    record = {
        "timestamp": datetime.utcnow().isoformat(),
        "question": question[:200],
        "confidence": round(confidence, 4),
        "escalated": escalated,
        "sources_used": sources_used,
        "vector_store": vector_store,
        "latency_s": round(latency_s, 3),
    }
    with open(METRICS_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

def load_metrics():
    """Load all records into a list of dicts."""
    if not os.path.exists(METRICS_FILE):
        return []
    records = []
    with open(METRICS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records