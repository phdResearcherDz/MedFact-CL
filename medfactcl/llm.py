"""Deterministic JSON generation with a local LLM served by Ollama."""
import json
import re
import time
from typing import Any, Optional, Type

from pydantic import BaseModel, ValidationError

DEFAULT_LLM = "gemma3:12b"


def _extract_json(text: str) -> Any:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"(\{.*\}|\[.*\])", text, flags=re.S)
        if m:
            return json.loads(m.group(1))
        raise


class OllamaJSON:
    """Calls an Ollama model at temperature 0 and validates the answer against a Pydantic schema.

    The JSON schema of the Pydantic model is passed to Ollama as a structured-output
    constraint, so the reply is always parseable; it is validated again on our side.
    """

    def __init__(self, model: str = DEFAULT_LLM, host: Optional[str] = None,
                 temperature: float = 0.0, num_ctx: int = 8192, seed: int = 42):
        import ollama
        self.client = ollama.Client(host=host) if host else ollama.Client()
        self.model = model
        self.options = {"temperature": temperature, "num_ctx": num_ctx, "seed": seed}

    def __call__(self, prompt: str, schema: Type[BaseModel], system: Optional[str] = None,
                 retries: int = 3) -> Optional[BaseModel]:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        last_err: Optional[Exception] = None
        for attempt in range(retries):
            try:
                resp = self.client.chat(model=self.model, messages=messages,
                                        format=schema.model_json_schema(), options=self.options)
                content = resp["message"]["content"]
                return schema.model_validate(_extract_json(content))
            except (ValidationError, json.JSONDecodeError, KeyError) as e:
                last_err = e
            except Exception as e:
                last_err = e
                time.sleep(1.0 + attempt)
        print(f"[llm] giving up after {retries} attempts: {last_err}")
        return None

