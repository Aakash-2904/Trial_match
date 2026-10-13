"""Local embeddings through Ollama (patient text never leaves the machine). Cached per text."""
import json
from typing import List

import numpy as np

import cache
from config import EMBED_MODEL

# nomic-embed-text expects task prefixes
PREFIX = {"query": "search_query: ", "document": "search_document: "}


def embed(texts: List[str], kind: str = "document", batch: int = 16) -> np.ndarray:
    import ollama

    out, todo = [None] * len(texts), []
    for i, t in enumerate(texts):
        hit = cache.get("embed", json.dumps([EMBED_MODEL, kind, t]))
        if hit is not None:
            out[i] = hit
        else:
            todo.append(i)
    for start in range(0, len(todo), batch):
        idx = todo[start:start + batch]
        resp = ollama.embed(model=EMBED_MODEL, input=[PREFIX[kind] + texts[i][:6000] for i in idx])
        for i, vec in zip(idx, resp["embeddings"]):
            cache.put("embed", json.dumps([EMBED_MODEL, kind, texts[i]]), vec)
            out[i] = vec
    m = np.asarray(out, dtype=np.float32)
    return m / np.clip(np.linalg.norm(m, axis=1, keepdims=True), 1e-8, None)
