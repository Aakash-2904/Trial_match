"""Data contracts for every agent. The two privacy tiers are separate types on purpose."""
from typing import List, Literal, Optional

from pydantic import BaseModel, Field, field_validator

from utils import clean_markdown


# ---------- privacy tiers ----------
class PublicTrialText(BaseModel):
    """PUBLIC tier: trial registry text. The only type llm_public will accept."""
    nct_id: str
    title: str
    conditions: List[str] = []
    summary: str = ""
    eligibility_text: str = ""


class PatientNote(BaseModel):
    """SENSITIVE tier: must only ever be processed by local models."""
    patient_id: str
    text: str


# ---------- trials ----------
class Trial(BaseModel):
    nct_id: str
    title: str
    conditions: List[str] = []
    summary: str = ""
    eligibility_text: str = ""
    min_age: Optional[float] = None
    max_age: Optional[float] = None
    sex: str = "ALL"
    status: str = ""

    def public(self) -> PublicTrialText:
        return PublicTrialText(nct_id=self.nct_id, title=self.title, conditions=self.conditions,
                               summary=self.summary, eligibility_text=self.eligibility_text)


class NumericCheck(BaseModel):
    field: str
    op: Literal["<", "<=", ">", ">=", "=="]
    value: float
    unit: Optional[str] = None


class Criterion(BaseModel):
    id: str
    kind: Literal["inclusion", "exclusion"]
    text: str
    category: Literal["age", "sex", "lab", "condition", "medication",
                      "procedure", "temporal", "other"] = "other"
    check: Optional[NumericCheck] = None

    @field_validator("text")
    @classmethod
    def _clean_text(cls, v: str) -> str:          # every parser path: local, API, imported batches
        return clean_markdown(v)


class ConceptCard(BaseModel):
    """Background knowledge written from PUBLIC trial text, retrieved by the checker."""
    term: str
    definition: str
    includes: List[str] = []


class TrialParseOutput(BaseModel):
    """What the API parser returns."""
    criteria: List[Criterion]
    search_terms: List[str] = []
    plain_summary: str = ""
    concepts: List[ConceptCard] = []


class ParsedTrial(BaseModel):
    nct_id: str
    criteria: List[Criterion]
    search_terms: List[str] = []
    plain_summary: str = ""
    concepts: List[ConceptCard] = []
    parser: Literal["api", "local"] = "local"


# ---------- patient ----------
class Fact(BaseModel):
    text: str
    evidence: str = Field(description="Exact span copied from the note")


class LabValue(BaseModel):
    name: str
    value: float
    unit: Optional[str] = None
    evidence: str


class PatientProfile(BaseModel):
    age: Optional[float] = None
    sex: Optional[Literal["male", "female"]] = None
    conditions: List[Fact] = []
    medications: List[Fact] = []
    procedures: List[Fact] = []
    labs: List[LabValue] = []
    negatives: List[Fact] = []
    other: List[Fact] = []


# ---------- agent outputs ----------
Truth = Literal["yes", "no", "unknown"]


class CriterionJudgment(BaseModel):
    id: str
    statement_true: Truth
    evidence: Optional[str] = None


class CheckerOutput(BaseModel):
    judgments: List[CriterionJudgment]


class CriticDecision(BaseModel):
    id: str
    verdict: Literal["confirm", "overturn"]
    corrected: Truth
    evidence: Optional[str] = None
    reason: str


class CriticOutput(BaseModel):
    decisions: List[CriticDecision]


class SingleCallOutput(BaseModel):
    label: Literal["eligible", "excluded", "not_relevant"]
    score: int = Field(ge=0, le=100)


# ---------- final results ----------
class CriterionResult(BaseModel):
    criterion: Criterion
    statement_true: Truth
    outcome: Literal["met", "not_met", "unknown"]   # from the eligibility point of view
    evidence: Optional[str] = None
    source: Literal["rule", "llm", "critic", "grounding", "cap"]
    note: Optional[str] = None


class TrialMatch(BaseModel):
    nct_id: str
    title: str
    plain_summary: str = ""
    label: Literal["likely_eligible", "possibly_eligible", "likely_ineligible"]
    eligibility_score: float
    relevance_score: float
    final_score: float
    explanation: str
    results: List[CriterionResult]
