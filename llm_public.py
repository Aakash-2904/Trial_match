"""PUBLIC tier: the ONLY module in this project allowed to call a cloud API.

It accepts PublicTrialText and nothing else. Patient data cannot reach this module:
the type guard rejects PatientNote, strings, dicts and every other type, and
tests/test_privacy.py checks that no other file imports a cloud SDK.
"""
import json
import time

from pydantic import BaseModel, ValidationError

import cache
from config import GEMINI_API_KEY, PUBLIC_MODEL
from schemas import PatientNote, PublicTrialText


class PrivacyViolation(TypeError):
    pass


def _guard(payload) -> None:
    if isinstance(payload, PatientNote) or not isinstance(payload, PublicTrialText):
        raise PrivacyViolation(
            f"llm_public only accepts PublicTrialText, got {type(payload).__name__}")


def available() -> bool:
    return bool(GEMINI_API_KEY)


def generate_json(payload: PublicTrialText, instruction: str, schema: type[BaseModel],
                  namespace: str, retries: int = 2) -> BaseModel:
    _guard(payload)                      # always first, before cache or network

    key = json.dumps([PUBLIC_MODEL, instruction, payload.model_dump()], sort_keys=True)
    hit = cache.get(namespace, key)
    if hit is not None:
        return schema.model_validate(hit)
    if not available():
        raise RuntimeError("GEMINI_API_KEY is not set")

    from google import genai
    from google.genai import types

    client = genai.Client(api_key=GEMINI_API_KEY)
    prompt = (f"{instruction}\n\nReturn JSON matching this schema:\n"
              f"{json.dumps(schema.model_json_schema())}\n\nTRIAL:\n{payload.model_dump_json(indent=1)}")

    for attempt in range(retries + 3):
        try:
            resp = client.models.generate_content(
                model=PUBLIC_MODEL, contents=prompt,
                config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0))
        except Exception as e:                       # rate limits: back off and retry
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                time.sleep(20 * (attempt + 1))
                continue
            raise
        try:
            obj = schema.model_validate_json(resp.text)
            cache.put(namespace, key, obj.model_dump())
            return obj
        except ValidationError as e:
            prompt += f"\n\nYour previous output was invalid: {e.errors()[:3]}. Return valid JSON only."
    raise RuntimeError(f"API parse failed for {payload.nct_id}")
