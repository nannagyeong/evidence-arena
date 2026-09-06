"""Environment-backed service settings; secrets are never logged."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date
from functools import lru_cache

from dotenv import load_dotenv

from .paths import PATHS


load_dotenv(PATHS.project_root / ".env", override=False)


@dataclass(frozen=True)
class Settings:
    llm_provider: str = ""
    llm_model: str = ""
    gemini_api_key: str = ""
    llm_timeout_sec: float = 30.0
    llm_max_retries: int = 1
    llm_force_failure: bool = False
    service_snapshot_date: date = date(2025, 12, 30)
    development_cutoff_date: date = date(2024, 12, 30)
    prototype_notice: str = "본 프로토타입은 2025년 말까지 확보된 데이터를 기준으로 동작합니다."

    @property
    def llm_configured(self) -> bool:
        return bool(
            self.llm_provider.lower() == "google"
            and self.llm_model.strip()
            and self.gemini_api_key.strip()
        )

    @classmethod
    def from_environment(cls) -> "Settings":
        return cls(
            llm_provider=os.getenv("LLM_PROVIDER", "").strip(),
            llm_model=os.getenv("LLM_MODEL", "").strip(),
            gemini_api_key=os.getenv("GEMINI_API_KEY", "").strip(),
            llm_timeout_sec=float(os.getenv("LLM_TIMEOUT_SEC", "30")),
            llm_max_retries=1,
            llm_force_failure=os.getenv("EVIDENCE_ARENA_LLM_FORCE_FAILURE", "0") == "1",
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings.from_environment()
