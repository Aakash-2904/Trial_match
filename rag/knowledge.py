"""Knowledge RAG: concept cards (written from PUBLIC trial text) retrieved per criterion for the checker."""
import re
from typing import Dict, List

import numpy as np

from config import DATA_DIR, KB_MAX_CARDS, KB_MIN_SIM
from rag.embed import embed
from rag.vector_store import VectorStore
from schemas import ConceptCard, Criterion, ParsedTrial
from utils import norm

KB_STORE = DATA_DIR / "knowledge_vectors"

ABBREVIATIONS = {"cv": "cardiovascular", "hf": "heart failure", "mi": "myocardial infarction", "htn": "hypertension",
                 "t2dm": "type 2 diabetes", "ckd": "chronic kidney disease"}
_ABBR = re.compile(r"\b(" + "|".join(ABBREVIATIONS) + r")\b")


def expand_abbreviations(text: str) -> str:
    """Normalised text plus the long forms of any known abbreviations, so 'HF' finds the 'heart failure' card."""
    t = norm(text)
    extra = dict.fromkeys(ABBREVIATIONS[m] for m in _ABBR.findall(t))
    return f"{t} ({'; '.join(extra)})" if extra else t


def card_text(c: ConceptCard) -> str:
    inc = f" Includes: {', '.join(c.includes)}." if c.includes else ""
    return f"{c.term}: {c.definition}{inc}"


def collect_cards(parsed: Dict[str, ParsedTrial]) -> List[ConceptCard]:
    seen: Dict[str, ConceptCard] = {}
    for p in parsed.values():
        for c in p.concepts:
            seen.setdefault(norm(c.term), c)          # first card per term wins
    return list(seen.values())


def build_knowledge_store(parsed: Dict[str, ParsedTrial]) -> VectorStore:
    cards = collect_cards(parsed)
    m = embed([card_text(c) for c in cards], kind="document") if cards else np.zeros((0, 1), np.float32)
    store = VectorStore([norm(c.term) for c in cards], m, [c.model_dump() for c in cards])
    store.save(KB_STORE)
    return store


class KnowledgeBase:
    def __init__(self, store: VectorStore):
        self.store = store

    def for_criteria(self, criteria: List[Criterion]) -> List[ConceptCard]:
        """Exact term matches first, then dense matches above a similarity threshold."""
        chosen: Dict[str, ConceptCard] = {}
        texts = [expand_abbreviations(c.text) for c in criteria
                 if c.category in ("condition", "medication", "procedure", "other")]
        for term, payload in zip(self.store.ids, self.store.payloads):
            if any(term and term in t for t in texts):
                chosen[term] = ConceptCard(**payload)
        if texts and len(chosen) < KB_MAX_CARDS:
            for qvec in embed(texts, kind="query"):
                for term, sim, payload in self.store.search(qvec, k=1):
                    if sim >= KB_MIN_SIM:
                        chosen.setdefault(term, ConceptCard(**payload))
        return list(chosen.values())[:KB_MAX_CARDS]
