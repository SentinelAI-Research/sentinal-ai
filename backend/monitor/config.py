from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_THRESHOLDS_PATH = ROOT / "configs" / "monitor_thresholds.yaml"


def load_monitor_thresholds(
    config_name: str = "C4",
    thresholds_path: Path | str = DEFAULT_THRESHOLDS_PATH,
) -> dict[str, float]:
    path = Path(thresholds_path)
    if not path.exists():
        return {"warn_threshold": 0.35, "block_threshold": 0.85}

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    warn_t = float(raw.get("warn_threshold", 0.35))
    block_t = float(raw.get("block_threshold", 0.85))
    cfg_overrides = (raw.get("configs") or {}).get(config_name.upper()) or {}
    if "warn_threshold" in cfg_overrides:
        warn_t = float(cfg_overrides["warn_threshold"])
    if "block_threshold" in cfg_overrides:
        block_t = float(cfg_overrides["block_threshold"])
    return {"warn_threshold": warn_t, "block_threshold": block_t}


@dataclass(frozen=True)
class MonitorConfig:
    """
    Unified configuration flags switching parts of one SentinelAI monitor on/off (B5):
    - C0:  No monitor (always ALLOW; logs ground truth)
    - C1:  Current action only (no sensitivity inheritance; derived objects reset to PUBLIC)
    - C2:  Action + trajectory history signals (still no lineage inheritance)
    - C2b: Session taint baseline (once CONFIDENTIAL is read in session, BLOCK any external send)
    - C3:  ML/heuristic risk model without provenance features (C2 + non-provenance risk)
    - C4:  Full SentinelAI (lineage inheritance, provenance features, trajectory, risk fusion, policy)
    """

    name: str = "C4"
    enabled: bool = True
    inherit_sensitivity: bool = True
    memory_preserves_sensitivity: bool = True
    use_trajectory_features: bool = True
    use_provenance_features: bool = True
    use_transformation_depth_signal: bool = True
    session_taint_mode: bool = False
    use_risk_fusion: bool = True
    use_policy_rules: bool = True
    warn_threshold: float = 0.35
    block_threshold: float = 0.85
    risk_model_path: str | None = None

    @classmethod
    def from_preset(
        cls,
        preset: str,
        *,
        warn_threshold: float | None = None,
        risk_model_path: str | None = None,
        thresholds_path: Path | str = DEFAULT_THRESHOLDS_PATH,
    ) -> "MonitorConfig":
        key = preset.strip()
        upper = key.upper()
        if upper == "C2B":
            canonical = "C2b"
        else:
            canonical = upper

        loaded = load_monitor_thresholds(canonical, thresholds_path=thresholds_path)
        effective_warn = (
            float(warn_threshold)
            if warn_threshold is not None
            else loaded["warn_threshold"]
        )
        effective_block = loaded["block_threshold"]

        if canonical == "C0":
            return cls(
                name="C0",
                enabled=False,
                inherit_sensitivity=True,
                memory_preserves_sensitivity=True,
                use_trajectory_features=False,
                use_provenance_features=False,
                use_transformation_depth_signal=False,
                session_taint_mode=False,
                use_risk_fusion=False,
                use_policy_rules=False,
                warn_threshold=1.01,
                block_threshold=1.01,
            )
        if canonical == "C1":
            return cls(
                name="C1",
                enabled=True,
                inherit_sensitivity=False,
                memory_preserves_sensitivity=False,
                use_trajectory_features=False,
                use_provenance_features=False,
                use_transformation_depth_signal=False,
                session_taint_mode=False,
                use_risk_fusion=False,
                use_policy_rules=True,
                warn_threshold=effective_warn,
                block_threshold=effective_block,
            )
        if canonical == "C2":
            return cls(
                name="C2",
                enabled=True,
                inherit_sensitivity=False,
                memory_preserves_sensitivity=False,
                use_trajectory_features=True,
                use_provenance_features=False,
                use_transformation_depth_signal=False,
                session_taint_mode=False,
                use_risk_fusion=True,
                use_policy_rules=True,
                warn_threshold=effective_warn,
                block_threshold=effective_block,
            )
        if canonical == "C2b":
            return cls(
                name="C2b",
                enabled=True,
                inherit_sensitivity=False,
                memory_preserves_sensitivity=False,
                use_trajectory_features=True,
                use_provenance_features=False,
                use_transformation_depth_signal=False,
                session_taint_mode=True,
                use_risk_fusion=False,
                use_policy_rules=True,
                warn_threshold=effective_warn,
                block_threshold=effective_block,
            )
        if canonical == "C3":
            return cls(
                name="C3",
                enabled=True,
                inherit_sensitivity=False,
                memory_preserves_sensitivity=False,
                use_trajectory_features=True,
                use_provenance_features=False,
                use_transformation_depth_signal=False,
                session_taint_mode=False,
                use_risk_fusion=True,
                use_policy_rules=True,
                warn_threshold=effective_warn,
                block_threshold=effective_block,
                risk_model_path=risk_model_path,
            )
        if canonical == "C4":
            return cls(
                name="C4",
                enabled=True,
                inherit_sensitivity=True,
                memory_preserves_sensitivity=True,
                use_trajectory_features=True,
                use_provenance_features=True,
                use_transformation_depth_signal=True,
                session_taint_mode=False,
                use_risk_fusion=True,
                use_policy_rules=True,
                warn_threshold=effective_warn,
                block_threshold=effective_block,
                risk_model_path=risk_model_path,
            )
        raise ValueError(
            f"Unknown MonitorConfig preset '{preset}'. Expected one of: C0, C1, C2, C2b, C3, C4."
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "enabled": self.enabled,
            "inherit_sensitivity": self.inherit_sensitivity,
            "memory_preserves_sensitivity": self.memory_preserves_sensitivity,
            "use_trajectory_features": self.use_trajectory_features,
            "use_provenance_features": self.use_provenance_features,
            "use_transformation_depth_signal": self.use_transformation_depth_signal,
            "session_taint_mode": self.session_taint_mode,
            "use_risk_fusion": self.use_risk_fusion,
            "use_policy_rules": self.use_policy_rules,
            "warn_threshold": self.warn_threshold,
            "block_threshold": self.block_threshold,
            "risk_model_path": self.risk_model_path,
        }
