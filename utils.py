import re

_TOKEN = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")
_NUMBER = re.compile(r"\d+(?:\.\d+)?")
_PLACEHOLDER = re.compile(r"\[\*\*.*?\*\*\]")          # MIMIC-style de-identification tags like [**2137**]
NEGATION = re.compile(r"\b(no|not|denies|denied|without|negative|absent|never|free of|ruled out)\b", re.I)
_SCOPE_END = re.compile(r"\.(?!\d)|[;:\n]|\b(?:but|which|however|although|except)\b", re.I)
_NEG_WINDOW = 6                                         # words on each side of a negation cue (NegEx-style scope)
_STOP = {"the", "and", "with", "for", "was", "has", "had", "of", "in", "on", "a", "an", "to", "history", "patient"}


def clean_markdown(s: str) -> str:
    """Drop markdown escapes from registry text: 'hemoglobin\\<9.5' -> 'hemoglobin<9.5'."""
    return re.sub(r"\\([^\w\s])", r"\1", s or "")


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").lower()).strip()


def canon(s: str) -> str:
    """Canonical key for matching field names: 'Ejection Fraction' == 'ejection_fraction'."""
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def numbers_in(s: str) -> set:
    return {f"{float(n):g}" for n in _NUMBER.findall(s or "")}


def grounded(evidence: str, source_text: str, min_overlap: float = 0.8) -> bool:
    """Is this evidence really supported by the source text?

    Small local models often paraphrase a quote slightly, so exact matching throws away good facts.
    Rule: (1) every number in the evidence must appear in the source (blocks invented values), and
    (2) at least `min_overlap` of the evidence's words must appear in the source.
    """
    if not evidence:
        return False
    ev, src = norm(_PLACEHOLDER.sub(" ", evidence)), norm(source_text)
    if len(re.findall(r"[a-z]", ev)) < 3:              # placeholders or bare numbers are not evidence
        return False
    if re.search(rf"(?<![a-z0-9]){re.escape(ev)}(?![a-z0-9])", src):   # whole words only: "ast" is not in "past"
        return True
    if not numbers_in(ev) <= numbers_in(src):
        return False
    ev_tokens = [t for t in _TOKEN.findall(ev) if len(t) > 2 or t.isdigit()]
    if not ev_tokens:
        return False
    src_tokens = set(_TOKEN.findall(src))
    return sum(t in src_tokens for t in ev_tokens) / len(ev_tokens) >= min_overlap


def outcome_for(kind: str, truth: str) -> str:
    """Translate 'is the criterion statement true?' into an eligibility outcome."""
    if truth == "unknown":
        return "unknown"
    if kind == "inclusion":
        return "met" if truth == "yes" else "not_met"
    return "not_met" if truth == "yes" else "met"


def supports(fact_text: str, evidence: str) -> bool:
    """Does the evidence actually talk about the fact? (shares at least one meaningful word)"""
    f = {t for t in _TOKEN.findall(norm(fact_text)) if t not in _STOP and len(t) > 2}
    e = set(_TOKEN.findall(norm(_PLACEHOLDER.sub(" ", evidence or ""))))
    return not f or any(_same_word(t, e) for t in f)


def _same_word(t: str, words) -> bool:
    return t in words or (len(t) >= 4 and any(t[:4] == x[:4] for x in words))


def negates(fact_text: str, evidence: str) -> bool:
    """Does the evidence explicitly deny this fact? A word of the fact must sit within a few words of a
    negation cue in the same clause, so "negative for any involvement which made this a primary cardiac
    lymphoma" negates involvement, not lymphoma."""
    f = {t for t in _TOKEN.findall(norm(NEGATION.sub(" ", fact_text or ""))) if t not in _STOP and len(t) > 2}
    if not f:
        return False
    for clause in _SCOPE_END.split(norm(_PLACEHOLDER.sub(" ", evidence or ""))):
        toks = _TOKEN.findall(clause)
        for m in NEGATION.finditer(clause):
            start, end = len(_TOKEN.findall(clause[:m.start()])), len(_TOKEN.findall(clause[:m.end()]))
            window = toks[max(0, start - _NEG_WINDOW):start] + toks[end:end + _NEG_WINDOW]
            if any(_same_word(t, window) for t in f):
                return True
    return False
