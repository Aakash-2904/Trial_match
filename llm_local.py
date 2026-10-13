"""LOCAL tier: the only LLM client allowed to see patient data. Runs through Ollama on this machine."""
import json

from pydantic import BaseModel, ValidationError

import cache
from config import LOCAL_CTX, LOCAL_MAX_TOKENS, LOCAL_MODEL, LOCAL_THREADS


def chat_json(system: str, user: str, schema: type[BaseModel], namespace: str,
              model: str = LOCAL_MODEL, retries: int = 2) -> BaseModel:
    """Ask the local model for JSON matching `schema`. Cached on disk; retries with the validation error."""
    import ollama  # imported lazily so tests run without Ollama installed

    key = json.dumps([model, system, user, schema.__name__])
    hit = cache.get(namespace, key)
    if hit is not None:
        return schema.model_validate(hit)

    base = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    messages = list(base)
    options = {"temperature": 0, "num_ctx": LOCAL_CTX, "num_thread": LOCAL_THREADS,
               "num_predict": LOCAL_MAX_TOKENS}    # greedy JSON decoding can loop forever without a cap
    last_err = None
    for _ in range(retries + 1):
        resp = ollama.chat(
            model=model,
            messages=messages,
            format=schema.model_json_schema(),     # constrained JSON decoding
            options=options,
            keep_alive="30m",                      # keep the model loaded between calls
        )
        raw = resp["message"]["content"]
        if resp.get("done_reason") == "length":    # hit the cap: the model was repeating itself
            last_err = f"output cut off at {LOCAL_MAX_TOKENS} tokens"
            print(f"    [llm_local] {namespace}: {last_err}, retrying with repeat_penalty", flush=True)
            messages, options = list(base), {**options, "repeat_penalty": 1.3}
            continue
        try:
            obj = schema.model_validate_json(raw)
            cache.put(namespace, key, obj.model_dump())
            return obj
        except ValidationError as e:
            last_err = e
            messages += [{"role": "assistant", "content": raw},
                         {"role": "user", "content": f"Invalid JSON for the schema: {e.errors()[:3]}. "
                                                     "Return corrected JSON only."}]
    raise RuntimeError(f"Local model did not return valid JSON: {last_err}")
