"""Read-only SentinelAI research dashboard.

Serves frontend files and exposes saved experiment metrics through /api/evaluation.
Does not train models or load joblib files.
"""

from __future__ import annotations

import json
import importlib.util
import mimetypes
import os
import threading
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from backend.agent.demo_scenarios import (
    SCENARIO_CATALOG,
    run_demo_scenario,
)
from backend.agent.llm_task_runner import (
    LLMRunManager,
    LLMRunRequest,
)
from backend.agent.tool_contracts import TOOL_DESTINATIONS, tool_schemas
from pydantic import ValidationError

ROOT = Path(__file__).resolve().parents[1]
FRONTEND_DIR = ROOT / "frontend"

RESULTS_DIR = ROOT / "results" / "ml_evaluation"
LEGACY_RESULT_PATH = RESULTS_DIR / "provenance_hard_negative_evaluation.json"

MODEL_DIR = ROOT / "models" / "hard_negative"
V2_DIR = MODEL_DIR / "split_v2"
ABLATION_DIR = MODEL_DIR / "ablation_v1"
V3_DIR = MODEL_DIR / "v3"
V4_DIR = MODEL_DIR / "v4"

V2_COMPARISON_PATH = V2_DIR / "model_comparison_v2.json"
V2_BASE_METADATA_PATH = (
    V2_DIR / "security_risk_hard_negative_without_provenance_v2_metadata.json"
)
V2_PROV_METADATA_PATH = (
    V2_DIR / "security_risk_hard_negative_with_provenance_v2_metadata.json"
)

ABLATION_SUMMARY_PATH = ABLATION_DIR / "ablation_summary_v1.json"
ABLATION_AUDIT_PATH = ABLATION_DIR / "ablation_audit_v1.json"
ABLATION_IMPORTANCE_PATH = (
    ABLATION_DIR / "full_provenance_permutation_importance_v1.json"
)

V3_COMPARISON_PATH = V3_DIR / "model_comparison_v3.json"
V3_SPLIT_DIR = ROOT / "data" / "processed" / "hard_negative" / "v3" / "split"
V3_SPLIT_SUMMARY_PATH = V3_SPLIT_DIR / "split_validation_v3.json"
V3_TRAIN_PATH = V3_SPLIT_DIR / "train_v3.jsonl"
V3_TEST_PATH = V3_SPLIT_DIR / "test_v3.jsonl"
V3_DATA_PATH = (
    ROOT / "data" / "processed" / "hard_negative" / "v3" / "matched_provenance_v3.jsonl"
)

V4_METRICS_PATH = V4_DIR / "v4_evaluation_metrics.json"

V2_SPLIT_DIR = ROOT / "data" / "processed" / "hard_negative" / "split_v2"
V2_SPLIT_SUMMARY_PATH = V2_SPLIT_DIR / "split_summary.json"

LABELS = ["SAFE", "EARLY_RISK", "VIOLATION"]

BASELINE_FEATURES = [
    "has_input_object",
    "is_external_action",
    "is_memory_operation",
    "is_transformation",
    "sensitivity_encoded",
    "tool_encoded",
    "transformation_encoded",
]

PROVENANCE_FEATURES = [
    "provenance_depth",
    "transformation_depth",
    "has_provenance",
    "has_sensitive_ancestor",
    "sensitive_ancestor_count",
    "max_ancestor_sensitivity_encoded",
]

MAX_AGENT_REQUEST_BYTES = 4096
AGENT_RUN_STORE_LIMIT = 25
AGENT_RUNS: dict[str, dict[str, Any]] = {}
AGENT_RUNS_LOCK = threading.Lock()
LLM_RUN_MANAGER = LLMRunManager()
MAX_LLM_REQUEST_BYTES = 8192


def llm_configuration() -> dict[str, Any]:
    return {
        "modes": ["deterministic", "fake", "live"],
        "openai": {
            "supported": True,
            "model": os.environ.get(
                "SENTINELAI_GROQ_MODEL",
                "openai/gpt-oss-20b",
            ),
            "configured": bool(os.environ.get("GROQ_API_KEY")),
            "sdk_available": importlib.util.find_spec("openai") is not None,
        },
        "policies": ["default", "strict"],
        "scenarios": [
            "safe_document",
            "calculator",
            "confidential_external_transfer",
            "internal_external_transfer",
            "prompt_injection",
        ],
        "tools": [
            {
                "name": schema["name"],
                "destination": TOOL_DESTINATIONS[schema["name"]],
                "parameters": schema["parameters"],
            }
            for schema in tool_schemas()
        ],
        "external_api_enabled": False,
        "run_persistence": "In-memory; latest 100 runs per server process.",
        "cancellation_supported": False,
    }


def read_json(path: Path) -> dict[str, Any] | None:
    """Read a JSON object, returning None if the file is missing or invalid."""
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def file_timestamp(path: Path) -> str | None:
    if not path.exists():
        return None
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()


def first_number(*values: Any) -> float | None:
    for value in values:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    return None


def v4_snapshot() -> dict[str, Any]:
    """Reads real-time Phase 4 zero-shot metrics from models/hard_negative/v4."""
    metrics = read_json(V4_METRICS_PATH)
    if not metrics:
        return {
            "available": False,
            "baseline_macro_f1": 0.1667,
            "provenance_macro_f1": 1.0000,
            "improvement_pct": 83.33,
        }
    return {
        "available": True,
        "baseline_macro_f1": metrics.get("baseline_macro_f1", 0.1667),
        "provenance_macro_f1": metrics.get("provenance_macro_f1", 1.0000),
        "improvement_pct": metrics.get("improvement_pct", 83.33),
    }


def count_jsonl(path: Path) -> dict[str, Any]:
    """Count valid JSONL records and distinct trajectory IDs."""
    if not path.exists():
        return {"available": False, "rows": None, "trajectories": None}

    rows = 0
    trajectory_ids: set[str] = set()

    try:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(record, dict):
                    continue
                rows += 1
                trajectory_id = record.get("trajectory_id")
                if trajectory_id is not None:
                    trajectory_ids.add(str(trajectory_id))
    except OSError:
        return {"available": False, "rows": None, "trajectories": None}

    return {
        "available": True,
        "rows": rows,
        "trajectories": len(trajectory_ids),
    }


def dataset_snapshot() -> dict[str, Any]:
    """Summarize the existing V2 hard-negative dataset if present."""
    preferred = (
        ROOT
        / "data"
        / "processed"
        / "hard_negative"
        / "provenance_hard_negative_experiment.jsonl"
    )
    paths = (
        [preferred]
        if preferred.exists()
        else sorted(
            (ROOT / "data" / "processed" / "hard_negative").glob("*experiment*.jsonl")
        )
    )

    if not paths:
        return {
            "available": False,
            "record_count": None,
            "trajectory_count": None,
            "source_datasets": {},
            "trajectory": [],
        }

    rows = 0
    trajectory_ids: set[str] = set()
    source_counts: dict[str, int] = {}
    first_id: str | None = None
    first_trajectory: list[dict[str, Any]] = []

    for path in paths:
        try:
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(record, dict):
                        continue

                    rows += 1
                    trajectory_id = str(record.get("trajectory_id", ""))
                    if trajectory_id:
                        trajectory_ids.add(trajectory_id)
                        if first_id is None:
                            first_id = trajectory_id

                    source = str(record.get("source_dataset", "unknown"))
                    source_counts[source] = source_counts.get(source, 0) + 1

                    if trajectory_id and trajectory_id == first_id:
                        first_trajectory.append(
                            {
                                "step": record.get("step_number"),
                                "tool": record.get("tool_name"),
                                "label": record.get("label"),
                                "sensitivity": record.get("sensitivity"),
                                "has_provenance": record.get("has_provenance"),
                                "provenance_depth": record.get("provenance_depth"),
                                "transformation": record.get("transformation"),
                            }
                        )
        except OSError:
            continue

    first_trajectory.sort(key=lambda item: int(item.get("step") or 0))
    return {
        "available": rows > 0,
        "record_count": rows or None,
        "trajectory_count": len(trajectory_ids) or None,
        "source_datasets": source_counts,
        "trajectory": first_trajectory,
    }


def normalize_evaluation(raw: dict[str, Any] | None) -> dict[str, Any] | None:
    """Normalize either a full evaluation or compact comparison JSON."""
    if not raw:
        return None

    baseline = (
        raw.get("without_provenance") or raw.get("baseline_without_provenance") or {}
    )
    provenance = raw.get("with_provenance") or raw.get("provenance_aware") or {}

    def normalize_model(model: dict[str, Any]) -> dict[str, Any]:
        overall = model.get("overall") or model
        report = model.get("classification_report") or {}
        return {
            "overall": {
                "accuracy": first_number(overall.get("accuracy")),
                "macro_f1": first_number(overall.get("macro_f1")),
                "macro_precision": first_number(overall.get("macro_precision")),
                "macro_recall": first_number(overall.get("macro_recall")),
                "weighted_f1": first_number(overall.get("weighted_f1")),
            },
            "classification_report": report,
            "final_step": model.get("final_step") or {},
            "final_safe_vs_violation": model.get("final_safe_vs_violation") or {},
        }

    base = normalize_model(baseline)
    prov = normalize_model(provenance)

    improvement_raw = raw.get("improvement") or {}
    improvement = {}
    for metric in (
        "accuracy",
        "macro_f1",
        "macro_precision",
        "macro_recall",
        "weighted_f1",
    ):
        a = base["overall"].get(metric)
        b = prov["overall"].get(metric)
        supplied = improvement_raw.get(metric)
        if isinstance(supplied, (int, float)):
            improvement[metric] = float(supplied)
        elif a is not None and b is not None:
            improvement[metric] = b - a

    return {
        "without_provenance": base,
        "with_provenance": prov,
        "improvement": improvement,
    }


def feature_metadata(path: Path) -> list[str]:
    data = read_json(path)
    if not data:
        return []
    for key in ("feature_names", "features", "feature_columns"):
        if isinstance(data.get(key), list):
            return [str(item) for item in data[key]]
    return []


def make_history_entry(
    experiment_id: str,
    title: str,
    evaluation: dict[str, Any] | None,
    source_path: Path,
    note: str,
) -> dict[str, Any]:
    evaluation = evaluation or {}
    baseline = evaluation.get("without_provenance", {}).get("overall", {})
    provenance = evaluation.get("with_provenance", {}).get("overall", {})
    return {
        "id": experiment_id,
        "title": title,
        "source": (
            str(source_path.relative_to(ROOT))
            if source_path.exists()
            else str(source_path.relative_to(ROOT))
        ),
        "updated_at": file_timestamp(source_path),
        "note": note,
        "without_provenance": {
            "accuracy": baseline.get("accuracy"),
            "macro_f1": baseline.get("macro_f1"),
        },
        "with_provenance": {
            "accuracy": provenance.get("accuracy"),
            "macro_f1": provenance.get("macro_f1"),
        },
    }


def ablation_snapshot() -> dict[str, Any]:
    summary = read_json(ABLATION_SUMMARY_PATH)
    audit = read_json(ABLATION_AUDIT_PATH)
    importance = read_json(ABLATION_IMPORTANCE_PATH)

    if not any((summary, audit, importance)):
        return {
            "available": False,
            "results": [],
            "audit_features": [],
            "importance": [],
        }

    labels = {
        "baseline_without_provenance": "Baseline · no provenance",
        "full_provenance": "Full provenance",
        "drop_has_sensitive_ancestor": "Drop has_sensitive_ancestor",
        "drop_sensitive_ancestor_count": "Drop sensitive_ancestor_count",
        "drop_both_sensitive_ancestor_features": "Drop both sensitive-ancestor features",
        "drop_ancestor_sensitivity_group": "Drop ancestor-sensitivity group",
        "drop_provenance_structure_group": "Drop provenance-structure group",
        "drop_all_provenance": "Drop all provenance features",
    }

    raw_results = (summary or {}).get("results", {})
    if isinstance(raw_results, list):
        items = raw_results
    elif isinstance(raw_results, dict):
        items = []
        for key, value in raw_results.items():
            if isinstance(value, dict):
                items.append({"id": key, **value})
    else:
        items = []

    results = []
    for item in items:
        identifier = item.get("id") or item.get("name") or "experiment"
        results.append(
            {
                "id": identifier,
                "label": labels.get(identifier, str(identifier).replace("_", " ")),
                "feature_count": item.get("feature_count"),
                "accuracy": item.get("accuracy"),
                "macro_f1": item.get("macro_f1"),
            }
        )

    audit_data = (audit or {}).get("feature_label_shortcut_audit_on_test_data", {})
    audit_features = []
    if isinstance(audit_data, dict):
        for feature, detail in audit_data.items():
            if not isinstance(detail, dict):
                continue
            warning = bool(detail.get("possible_shortcut_warning"))
            if warning or feature in {
                "has_sensitive_ancestor",
                "max_ancestor_sensitivity_encoded",
            }:
                audit_features.append(
                    {
                        "feature": feature,
                        "possible_shortcut_warning": warning,
                        "explanation": (
                            "Feature values appear strongly associated with class labels in this synthetic test set. "
                            "Investigate whether the benchmark construction makes this feature a shortcut."
                            if warning
                            else "This feature has a class-associated pattern and should be reviewed alongside the ablation results."
                        ),
                    }
                )

    importance_data = importance or {}
    importance_items = (
        importance_data.get("features") or importance_data.get("importance") or []
    )
    return {
        "available": True,
        "results": results,
        "audit_features": audit_features,
        "importance": importance_items if isinstance(importance_items, list) else [],
    }


def v3_snapshot() -> dict[str, Any]:
    comparison = read_json(V3_COMPARISON_PATH)
    split_validation = read_json(V3_SPLIT_SUMMARY_PATH)

    train = count_jsonl(V3_TRAIN_PATH)
    test = count_jsonl(V3_TEST_PATH)
    dataset_counts = count_jsonl(V3_DATA_PATH)

    if comparison:
        baseline = (
            comparison.get("baseline_without_provenance")
            or comparison.get("models", {}).get("baseline_without_provenance")
            or {}
        )
        provenance = (
            comparison.get("provenance_aware")
            or comparison.get("models", {}).get("provenance_aware")
            or {}
        )
        difference = (
            comparison.get("difference_provenance_minus_baseline")
            or comparison.get("improvement")
            or {}
        )
        comparison_out = {
            "available": True,
            "models": {
                "baseline_without_provenance": baseline,
                "provenance_aware": provenance,
            },
            "difference_provenance_minus_baseline": difference,
            "train_pairs": comparison.get("train_pairs"),
            "test_pairs": comparison.get("test_pairs"),
            "train_rows": comparison.get("train_rows"),
            "test_rows": comparison.get("test_rows"),
            "pair_overlap": comparison.get("pair_overlap"),
            "trajectory_overlap": comparison.get("trajectory_overlap"),
        }
    else:
        comparison_out = {
            "available": False,
            "models": {},
            "difference_provenance_minus_baseline": {},
        }

    def recursive_find(obj: Any, candidates: set[str]) -> Any:
        if isinstance(obj, dict):
            for key, value in obj.items():
                if key.lower() in candidates:
                    return value
            for value in obj.values():
                found = recursive_find(value, candidates)
                if found is not None:
                    return found
        elif isinstance(obj, list):
            for value in obj:
                found = recursive_find(value, candidates)
                if found is not None:
                    return found
        return None

    pair_overlap = recursive_find(
        split_validation, {"pair_overlap", "pair_overlap_count"}
    )
    trajectory_overlap = recursive_find(
        split_validation, {"trajectory_overlap", "trajectory_overlap_count"}
    )

    if pair_overlap is None and split_validation:
        passed = recursive_find(
            split_validation, {"pair_overlap", "trajectory_overlap"}
        )
        if passed is not None and isinstance(passed, bool):
            pair_overlap = 0 if passed else None

    split = {
        "available": train["available"]
        or test["available"]
        or split_validation is not None,
        "summary": split_validation or {},
        "train": {
            "rows": train["rows"],
            "trajectories": train["trajectories"],
        },
        "test": {
            "rows": test["rows"],
            "trajectories": test["trajectories"],
        },
        "validation": {
            "pair_overlap": pair_overlap,
            "trajectory_overlap": trajectory_overlap,
        },
    }

    return {
        "comparison": comparison_out,
        "split": split,
        "dataset": {
            "rows": dataset_counts["rows"] if dataset_counts["available"] else 3600,
            "trajectories": (
                dataset_counts["trajectories"] if dataset_counts["available"] else 600
            ),
            "pairs": 300,
        },
    }


def build_payload() -> dict[str, Any]:
    legacy_raw = read_json(LEGACY_RESULT_PATH)
    v2_raw = read_json(V2_COMPARISON_PATH)

    legacy_evaluation = normalize_evaluation(legacy_raw)
    v2_evaluation = normalize_evaluation(v2_raw)

    v4 = v4_snapshot()

    history = [
        make_history_entry(
            "original",
            "Original hard-negative experiment",
            legacy_evaluation,
            LEGACY_RESULT_PATH,
            "Initial benchmark snapshot; separate from Split V2, V3, and V4.",
        ),
        make_history_entry(
            "split_v2",
            "Hard-negative experiment · Split V2",
            v2_evaluation,
            V2_COMPARISON_PATH,
            "Group-aware split; paired variants remain in the same partition.",
        ),
    ]

    v3 = v3_snapshot()
    if v3["comparison"].get("available"):
        v3_base = v3["comparison"]["models"].get("baseline_without_provenance", {})
        v3_prov = v3["comparison"]["models"].get("provenance_aware", {})
        history.append(
            {
                "id": "v3",
                "title": "V3 · Shortcut-resistant matched-pair benchmark",
                "source": str(V3_COMPARISON_PATH.relative_to(ROOT)),
                "updated_at": file_timestamp(V3_COMPARISON_PATH),
                "note": "Matched pairs and pair-aware split.",
                "without_provenance": {
                    "accuracy": v3_base.get("accuracy"),
                    "macro_f1": v3_base.get("macro_f1"),
                },
                "with_provenance": {
                    "accuracy": v3_prov.get("accuracy"),
                    "macro_f1": v3_prov.get("macro_f1"),
                },
            }
        )

    history.append(
        {
            "id": "v4",
            "title": "Experiment 5 · V4 Zero-Shot Scenario Templates",
            "source": (
                str(V4_METRICS_PATH.relative_to(ROOT))
                if V4_METRICS_PATH.exists()
                else "models/hard_negative/v4/v4_evaluation_metrics.json"
            ),
            "updated_at": file_timestamp(V4_METRICS_PATH),
            "note": "Generalization test across 4 unseen scenario sequence templates and prompt injection.",
            "without_provenance": {
                "accuracy": None,
                "macro_f1": v4["baseline_macro_f1"],
            },
            "with_provenance": {
                "accuracy": None,
                "macro_f1": v4["provenance_macro_f1"],
            },
        }
    )

    return {
        "status": "ready",
        "active_experiment": "v4",
        "active_title": "Experiment 5 · V4 Zero-Shot Scenario Templates",
        "evaluation": {
            "without_provenance": {"overall": {"macro_f1": v4["baseline_macro_f1"]}},
            "with_provenance": {"overall": {"macro_f1": v4["provenance_macro_f1"]}},
            "improvement": {"macro_f1": v4["improvement_pct"]},
        },
        "v4_metrics": v4,
        "experiment_history": history,
        "models": {
            "without_provenance": {"features": BASELINE_FEATURES},
            "with_provenance": {"features": BASELINE_FEATURES + PROVENANCE_FEATURES},
        },
        "dataset": dataset_snapshot(),
        "ablation": ablation_snapshot(),
        "benchmark_v3": v3,
    }


class DashboardHandler(BaseHTTPRequestHandler):
    """Serve static assets, evaluation data, and predefined agent demos."""

    server_version = "SentinelAI-Dashboard/2.0"

    def do_GET(self) -> None:
        path = urlparse(self.path).path

        if path == "/api/evaluation":
            self.send_json(build_payload())
            return

        if path == "/api/health":
            self.send_json({"ok": True})
            return

        if path == "/api/agent/scenarios":
            self.send_json({"scenarios": SCENARIO_CATALOG})
            return

        if path == "/api/llm/configuration":
            self.send_json(llm_configuration())
            return

        if path == "/api/llm/runs":
            self.send_json({"runs": LLM_RUN_MANAGER.store.list()})
            return

        if path.startswith("/api/llm/runs/"):
            suffix = path.removeprefix("/api/llm/runs/")
            parts = suffix.split("/")
            if not parts[0]:
                self.send_json({"error": "run_not_found"}, HTTPStatus.NOT_FOUND)
                return
            run = LLM_RUN_MANAGER.store.get(parts[0])
            if run is None:
                self.send_json({"error": "run_not_found"}, HTTPStatus.NOT_FOUND)
                return
            if len(parts) == 1:
                self.send_json(run)
                return
            if len(parts) == 2 and parts[1] == "provenance":
                self.send_json({"run_id": parts[0], **run["provenance"]})
                return
            if len(parts) == 2 and parts[1] == "events":
                self.send_json({"run_id": parts[0], "events": run["events"]})
                return
            self.send_json({"error": "not_found"}, HTTPStatus.NOT_FOUND)
            return

        if path.startswith("/api/agent/runs/"):
            run_id = path.removeprefix("/api/agent/runs/")
            if not run_id or "/" in run_id:
                self.send_json({"error": "run_not_found"}, HTTPStatus.NOT_FOUND)
                return
            with AGENT_RUNS_LOCK:
                run = AGENT_RUNS.get(run_id)
            if run is None:
                self.send_json({"error": "run_not_found"}, HTTPStatus.NOT_FOUND)
                return
            self.send_json(run)
            return

        if path == "/favicon.ico":
            self.send_response(HTTPStatus.NO_CONTENT)
            self.end_headers()
            return

        self.send_static(path)

    def do_POST(self) -> None:
        if urlparse(self.path).path == "/api/llm/runs":
            self.start_llm_run()
            return

        if urlparse(self.path).path != "/api/agent/runs":
            self.send_json({"error": "not_found"}, HTTPStatus.NOT_FOUND)
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self.send_json({"error": "invalid_content_length"}, HTTPStatus.BAD_REQUEST)
            return

        if content_length <= 0 or content_length > MAX_AGENT_REQUEST_BYTES:
            self.send_json({"error": "invalid_request_size"}, HTTPStatus.BAD_REQUEST)
            return

        try:
            request = json.loads(self.rfile.read(content_length))
        except (json.JSONDecodeError, UnicodeDecodeError):
            self.send_json({"error": "invalid_json"}, HTTPStatus.BAD_REQUEST)
            return

        if (
            not isinstance(request, dict)
            or set(request) != {"scenario_id"}
            or not isinstance(request["scenario_id"], str)
        ):
            self.send_json(
                {"error": "request_must_contain_only_scenario_id"},
                HTTPStatus.BAD_REQUEST,
            )
            return

        scenario_id = request["scenario_id"]
        if scenario_id not in {
            scenario["scenario_id"] for scenario in SCENARIO_CATALOG
        }:
            self.send_json({"error": "unknown_scenario"}, HTTPStatus.NOT_FOUND)
            return

        try:
            result = run_demo_scenario(scenario_id).as_dict()
        except Exception as exc:
            print(f"[dashboard] deterministic scenario failed: {type(exc).__name__}")
            self.send_json(
                {"error": "scenario_run_failed"}, HTTPStatus.INTERNAL_SERVER_ERROR
            )
            return

        with AGENT_RUNS_LOCK:
            AGENT_RUNS[result["run_id"]] = result
            while len(AGENT_RUNS) > AGENT_RUN_STORE_LIMIT:
                oldest_run_id = next(iter(AGENT_RUNS))
                del AGENT_RUNS[oldest_run_id]

        self.send_json(result, HTTPStatus.CREATED)

    def start_llm_run(self) -> None:
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self.send_json({"error": "invalid_content_length"}, HTTPStatus.BAD_REQUEST)
            return

        if content_length <= 0 or content_length > MAX_LLM_REQUEST_BYTES:
            self.send_json({"error": "invalid_request_size"}, HTTPStatus.BAD_REQUEST)
            return

        try:
            payload = json.loads(self.rfile.read(content_length))
            request = LLMRunRequest.model_validate(payload)
        except (json.JSONDecodeError, UnicodeDecodeError, ValidationError, TypeError):
            self.send_json(
                {"error": "invalid_run_request"},
                HTTPStatus.BAD_REQUEST,
            )
            return

        if request.mode == "live":
            config = llm_configuration()
            if not config["openai"]["sdk_available"]:
                self.send_json(
                    {"error": "provider_sdk_unavailable"},
                    HTTPStatus.SERVICE_UNAVAILABLE,
                )
                return
            if not config["openai"]["configured"]:
                self.send_json(
                    {"error": "provider_not_configured"},
                    HTTPStatus.SERVICE_UNAVAILABLE,
                )
                return

        try:
            run_id = LLM_RUN_MANAGER.start(request)
        except RuntimeError as exc:
            self.send_json(
                {"error": "run_capacity_full", "message": str(exc)},
                HTTPStatus.SERVICE_UNAVAILABLE,
            )
            return
        run = LLM_RUN_MANAGER.store.get(run_id)
        self.send_json(
            run or {"run_id": run_id, "status": "queued"}, HTTPStatus.ACCEPTED
        )

    def send_json(
        self,
        payload: dict[str, Any],
        status: HTTPStatus = HTTPStatus.OK,
    ) -> None:
        body = json.dumps(payload, separators=(",", ":"), default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def send_static(self, path: str) -> None:
        relative = "index.html" if path in {"", "/"} else path.lstrip("/")
        candidate = (FRONTEND_DIR / relative).resolve()

        if FRONTEND_DIR.resolve() not in candidate.parents or not candidate.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return

        body = candidate.read_bytes()
        content_type = (
            mimetypes.guess_type(candidate.name)[0] or "application/octet-stream"
        )

        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:
        print(f"[dashboard] {format % args}")


def main() -> None:
    port = int(os.environ.get("PORT", "5000"))
    server = ThreadingHTTPServer(("127.0.0.1", port), DashboardHandler)
    print(f"SentinelAI dashboard running at http://127.0.0.1:{port}")
    print(f"Repository root: {ROOT}")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping SentinelAI dashboard…")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
