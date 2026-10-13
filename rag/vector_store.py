"""Minimal persistent vector store: a normalised numpy matrix plus ids. Fine for thousands of docs on a laptop.
(Swapping in ChromaDB later only touches this file.)"""
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np


class VectorStore:
    def __init__(self, ids: List[str], matrix: np.ndarray, payloads: List[dict] = None):
        self.ids, self.matrix = ids, matrix
        self.payloads = payloads or [{} for _ in ids]

    def search(self, qvec: np.ndarray, k: int = 10) -> List[Tuple[str, float, dict]]:
        if not self.ids:
            return []
        sims = self.matrix @ qvec
        top = np.argsort(-sims)[:k]
        return [(self.ids[i], float(sims[i]), self.payloads[i]) for i in top]

    def similarities(self, qvec: np.ndarray) -> dict:
        return dict(zip(self.ids, (self.matrix @ qvec).tolist())) if self.ids else {}

    def save(self, path: Path):
        np.save(path.with_suffix(".npy"), self.matrix)
        path.with_suffix(".json").write_text(json.dumps({"ids": self.ids, "payloads": self.payloads}), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "VectorStore":
        meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        return cls(meta["ids"], np.load(path.with_suffix(".npy")), meta["payloads"])

    @staticmethod
    def exists(path: Path) -> bool:
        return path.with_suffix(".npy").exists() and path.with_suffix(".json").exists()
