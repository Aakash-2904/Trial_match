"""Hybrid trial retrieval: BM25 + dense embeddings, fused with Reciprocal Rank Fusion."""
from typing import Dict, List

from agents.prefilter import TrialIndex
from config import DATA_DIR, RRF_K
from rag.embed import embed
from rag.vector_store import VectorStore
from schemas import ParsedTrial, Trial

TRIAL_STORE = DATA_DIR / "trial_vectors"


def trial_text(t: Trial, p: ParsedTrial = None) -> str:
    terms = ", ".join(p.search_terms) if p else ""
    return f"{t.title}. Conditions: {', '.join(t.conditions)}. {terms}. {t.summary[:1500]}"


def build_trial_store(trials: List[Trial], parsed: Dict[str, ParsedTrial]) -> VectorStore:
    m = embed([trial_text(t, parsed.get(t.nct_id)) for t in trials], kind="document")
    store = VectorStore([t.nct_id for t in trials], m)
    store.save(TRIAL_STORE)
    return store


def rrf(rankings: List[List[str]], k: int = RRF_K) -> Dict[str, float]:
    fused: Dict[str, float] = {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking):
            fused[doc] = fused.get(doc, 0.0) + 1.0 / (k + rank + 1)
    return fused


class HybridRetriever:
    def __init__(self, trials: List[Trial], parsed: Dict[str, ParsedTrial], store: VectorStore):
        self.trials, self.parsed, self.store = trials, parsed, store
        self.by_id = {t.nct_id: t for t in trials}
        self.bm25 = TrialIndex(trials, parsed)

    def scores(self, query: str, restrict_to: List[str] = None) -> Dict[str, float]:
        """Fused scores normalised to 0..1, optionally only among a candidate set."""
        ids = set(restrict_to) if restrict_to else set(self.by_id)
        bm = {k: v for k, v in self.bm25.scores(query).items() if k in ids}
        dense = {k: v for k, v in self.store.similarities(embed([query], kind="query")[0]).items() if k in ids}
        fused = rrf([sorted(bm, key=bm.get, reverse=True), sorted(dense, key=dense.get, reverse=True)])
        top = max(fused.values(), default=0.0)
        return {k: (v / top if top else 0.0) for k, v in fused.items()}

    def search(self, query: str, k: int) -> List[Trial]:
        s = self.scores(query)
        return [self.by_id[n] for n in sorted(s, key=s.get, reverse=True)[:k]]
