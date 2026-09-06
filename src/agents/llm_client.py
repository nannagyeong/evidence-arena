"""Google Gemini Structured Output adapter with one retry and no tools."""

from __future__ import annotations

import json
import time

from src.config.settings import Settings

from .prompts import build_initial_prompt


class LLMUnavailable(RuntimeError):
    def __init__(self, message: str, error_code: str = "LLM_UNAVAILABLE"):
        super().__init__(message)
        self.error_code = error_code


def _safe_error_code(error: Exception | None) -> str:
    if error is None:
        return "UNKNOWN"
    status = getattr(error, "code", None) or getattr(error, "status_code", None)
    return f"{type(error).__name__}:{status}" if status is not None else type(error).__name__


def _claim_array_schema(claim_schema: dict) -> dict:
    return {"type": "array", "minItems": 1, "maxItems": 3, "items": claim_schema}


class GoogleStructuredClaimClient:
    def __init__(self, settings: Settings, telemetry: dict):
        self.settings = settings
        self.telemetry = telemetry

    def generate_structured(self, stage: str, prompt: str, payload: dict, response_schema: dict):
        if not self.settings.llm_configured:
            raise LLMUnavailable("LLM provider/model/key is not configured")

        from google import genai
        from google.genai import types

        contents = (
            prompt
            + "\n\n검증된 입력 JSON:\n"
            + json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=str)
        )
        last_error: Exception | None = None
        for attempt in range(self.settings.llm_max_retries + 1):
            started = time.perf_counter()
            self.telemetry["llm_calls"] += 1
            try:
                if self.settings.llm_force_failure:
                    raise RuntimeError("forced LLM failure for E2E evaluation")
                client = genai.Client(
                    api_key=self.settings.gemini_api_key,
                    http_options=types.HttpOptions(timeout=int(self.settings.llm_timeout_sec * 1000)),
                )
                response = client.models.generate_content(
                    model=self.settings.llm_model,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        temperature=0,
                        response_mime_type="application/json",
                        response_json_schema=response_schema,
                    ),
                )
                parsed = json.loads(response.text)
                self.telemetry["llm_latency_sec"] += time.perf_counter() - started
                return parsed
            except Exception as error:
                last_error = error
                self.telemetry["llm_latency_sec"] += time.perf_counter() - started
                if attempt >= self.settings.llm_max_retries:
                    break
                time.sleep(1.5 * (attempt + 1))
        error_code = _safe_error_code(last_error)
        raise LLMUnavailable(
            f"Gemini Structured Output failed at {stage}: {error_code}",
            error_code=error_code,
        ) from last_error

    def generate_claims(
        self,
        role: str,
        fact_room_view: dict,
        claim_schema: dict,
        allowed_claim_blueprints: list[dict],
    ) -> list[dict]:
        parsed = self.generate_structured(
            f"{role}_initial",
            build_initial_prompt(role),
            {
                "fact_room": fact_room_view,
                "allowed_claim_blueprints": allowed_claim_blueprints,
            },
            _claim_array_schema(claim_schema),
        )
        if not isinstance(parsed, list):
            raise ValueError("Structured Claim output must be a list")
        return parsed
