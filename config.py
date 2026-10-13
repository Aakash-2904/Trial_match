"""Central configuration. Values can be overridden in a .env file."""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

DATA_DIR = ROOT / "data"
CACHE_DIR = ROOT / ".cache"
RESULTS_DIR = ROOT / "results"

# Local tier (sees patient data)
LOCAL_MODEL = os.getenv("LOCAL_MODEL", "qwen2.5:3b")
LOCAL_CTX = int(os.getenv("LOCAL_CTX", "4096"))
LOCAL_THREADS = int(os.getenv("LOCAL_THREADS", "4"))
LOCAL_MAX_TOKENS = int(os.getenv("LOCAL_MAX_TOKENS", "2048"))   # cap on generated tokens per call
MAX_FUZZY_CRITERIA = int(os.getenv("MAX_FUZZY_CRITERIA", "30"))

# RAG (all local)
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
RRF_K = 60
KB_MIN_SIM = float(os.getenv("KB_MIN_SIM", "0.6"))
KB_MAX_CARDS = int(os.getenv("KB_MAX_CARDS", "6"))

# Public tier (trial registry text only)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
PUBLIC_MODEL = os.getenv("PUBLIC_MODEL", "gemini-2.5-flash")

# Evaluation size (keep small on a CPU laptop)
EVAL_PATIENTS = int(os.getenv("EVAL_PATIENTS", "10"))
EVAL_TRIALS_PER_PATIENT = int(os.getenv("EVAL_TRIALS_PER_PATIENT", "12"))
SEED = 42

TOPICS_URL = "https://trec.nist.gov/data/trials/topics2021.xml"
QRELS_URL = "https://trec.nist.gov/data/trials/qrels2021.txt"
CTGOV_STUDY_URL = "https://clinicaltrials.gov/api/v2/studies/{nct}"
CTGOV_SEARCH_URL = "https://clinicaltrials.gov/api/v2/studies"

# Shared vocabulary so the parser (API) and profiler (local) name measurements the same way
CANONICAL_FIELDS = [
    "age", "eGFR", "creatinine", "HbA1c", "glucose", "ejection_fraction", "BMI", "weight_kg",
    "hemoglobin", "platelets", "WBC", "ANC", "ALT", "AST", "bilirubin", "albumin", "INR",
    "systolic_bp", "diastolic_bp", "heart_rate", "LDL", "potassium", "sodium", "TSH", "PSA",
    "ECOG", "Karnofsky", "MMSE",
]
