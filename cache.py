"""Tiny disk cache so crashes and re-runs never repeat an LLM call."""
import hashlib
import json

from config import CACHE_DIR


def _path(namespace: str, key: str):
    d = CACHE_DIR / namespace
    d.mkdir(parents=True, exist_ok=True)
    return d / (hashlib.sha256(key.encode("utf-8")).hexdigest()[:32] + ".json")


def get(namespace: str, key: str):
    p = _path(namespace, key)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def put(namespace: str, key: str, value) -> None:
    _path(namespace, key).write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
