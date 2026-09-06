"""Portable paths for STEP15 runtime code."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    project_root: Path
    data_root: Path
    artifact_root: Path
    feature_dir: Path
    rag_dir: Path
    model_dir: Path
    claim_dir: Path
    routing_dir: Path
    verdict_dir: Path

    @classmethod
    def from_environment(cls) -> "ProjectPaths":
        project_root = Path(__file__).resolve().parents[2]
        data_value = os.getenv("DATA_ROOT", "").strip()
        artifact_value = os.getenv("ARTIFACT_ROOT", "").strip()
        data_root = Path(data_value).expanduser().resolve() if data_value else (project_root / "data").resolve()
        artifact_root = Path(artifact_value).expanduser().resolve() if artifact_value else (project_root / "artifacts").resolve()
        return cls(
            project_root=project_root,
            data_root=data_root,
            artifact_root=artifact_root,
            feature_dir=data_root / "features",
            rag_dir=data_root / "rag_evidence",
            model_dir=data_root / "ml_evidence" / "final",
            claim_dir=data_root / "claim_validator",
            routing_dir=data_root / "claim_routing",
            verdict_dir=data_root / "evidence_synthesis",
        )

    @property
    def feature_panel(self) -> Path:
        return self.feature_dir / "feature_panel.parquet"

    @property
    def feature_registry(self) -> Path:
        return self.feature_dir / "01_feature_registry.csv"

    @property
    def frozen_model(self) -> Path:
        return self.model_dir / "models" / "catboost_CAT_012_final_2019_2024.cbm"

    @property
    def model_contract(self) -> Path:
        return self.model_dir / "10_model_freeze_contract.json"

    @property
    def ml_adapter(self) -> Path:
        return self.rag_dir / "16_step10_ml_evidence_adapter_2025.parquet"


PATHS = ProjectPaths.from_environment()
