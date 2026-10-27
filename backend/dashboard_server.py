"""Read-only SentinelAI research dashboard.

Serves frontend files and exposes saved experiment metrics through /api/evaluation.
Does not train models or load joblib files.
"""

from __future__ import annotations

import json
import mimetypes
import os
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
FRONTEND_DIR = ROOT / "frontend"

RESULTS_DIR = ROOT / "results" / "ml_evaluation"
LEGACY_RESULT_PATH = RESULTS_DIR / "provenance_hard_negative_evaluation.json"

MODEL_DIR = ROOT / "models" / "hard_negative"
V2_DIR = MODEL_DIR / "split_v2"
ABLATION_DIR = MODEL_DIR / "ablation_v1"
V3_DIR = MODEL_DIR / "v3"

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
    active_evaluation = v2_evaluation or legacy_evaluation
    active_id = "split_v2" if v2_evaluation else "original"

    history = [
        make_history_entry(
            "original",
            "Original hard-negative experiment",
            legacy_evaluation,
            LEGACY_RESULT_PATH,
            "Initial benchmark snapshot; separate from Split V2 and V3.",
        ),
        make_history_entry(
            "split_v2",
            "Hard-negative experiment · Split V2",
            v2_evaluation,
            V2_COMPARISON_PATH,
            "Group-aware split; paired variants remain in the same partition. See the V2 ablation section for shortcut diagnostics.",
        ),
    ]

    v3 = v3_snapshot()
    v3_comparison = v3["comparison"]
    v3_models = v3_comparison.get("models", {})
    v3_baseline = v3_models.get("baseline_without_provenance", {})
    v3_provenance = v3_models.get("provenance_aware", {})

    if v3_comparison.get("available"):
        history.append(
            {
                "id": "v3",
                "title": "V3 · Shortcut-resistant matched-pair benchmark",
                "source": str(V3_COMPARISON_PATH.relative_to(ROOT)),
                "updated_at": file_timestamp(V3_COMPARISON_PATH),
                "note": "Matched pairs and pair-aware split; fixed action sequence remains a limitation.",
                "without_provenance": {
                    "accuracy": v3_baseline.get("accuracy"),
                    "macro_f1": v3_baseline.get("macro_f1"),
                },
                "with_provenance": {
                    "accuracy": v3_provenance.get("accuracy"),
                    "macro_f1": v3_provenance.get("macro_f1"),
                },
            }
        )

    if active_evaluation:
        baseline_overall = active_evaluation["without_provenance"]["overall"]
        provenance_overall = active_evaluation["with_provenance"]["overall"]
        models = {
            "without_provenance": {
                "features": feature_metadata(V2_BASE_METADATA_PATH)
                or BASELINE_FEATURES,
            },
            "with_provenance": {
                "features": feature_metadata(V2_PROV_METADATA_PATH)
                or BASELINE_FEATURES + PROVENANCE_FEATURES,
            },
        }
    else:
        models = {
            "without_provenance": {"features": BASELINE_FEATURES},
            "with_provenance": {"features": BASELINE_FEATURES + PROVENANCE_FEATURES},
        }

    # V3 uses the latest matched-pair comparison for the headline cards when available.
    if v3_comparison.get("available"):
        v3_eval = normalize_evaluation(
            {
                "without_provenance": v3_baseline,
                "with_provenance": v3_provenance,
                "improvement": {
                    "accuracy": (
                        v3_baseline.get("accuracy") is not None
                        and v3_provenance.get("accuracy") is not None
                    )
                    and v3_provenance["accuracy"] - v3_baseline["accuracy"],
                    "macro_f1": (
                        v3_baseline.get("macro_f1") is not None
                        and v3_provenance.get("macro_f1") is not None
                    )
                    and v3_provenance["macro_f1"] - v3_baseline["macro_f1"],
                },
            }
        )
        active_evaluation = v3_eval
        active_id = "v3"
        models = {
            "without_provenance": {"features": BASELINE_FEATURES},
            "with_provenance": {"features": BASELINE_FEATURES + PROVENANCE_FEATURES},
        }

    return {
        "status": "ready" if active_evaluation else "artifacts_unavailable",
        "active_experiment": active_id,
        "active_title": {
            "original": "Original hard-negative experiment",
            "split_v2": "Split V2 · Hard-negative experiment",
            "v3": "V3 · Shortcut-resistant matched-pair benchmark",
        }.get(active_id, active_id),
        "evaluation_source": {
            "original": str(LEGACY_RESULT_PATH.relative_to(ROOT)),
            "split_v2": str(V2_COMPARISON_PATH.relative_to(ROOT)),
            "v3": str(V3_COMPARISON_PATH.relative_to(ROOT)),
        }.get(active_id),
        "evaluation_updated_at": file_timestamp(
            {
                "original": LEGACY_RESULT_PATH,
                "split_v2": V2_COMPARISON_PATH,
                "v3": V3_COMPARISON_PATH,
            }[active_id]
        ),
        "evaluation": active_evaluation,
        "experiment_history": history,
        "models": models,
        "dataset": dataset_snapshot(),
        "ablation": ablation_snapshot(),
        "benchmark_v3": v3,
    }


class DashboardHandler(BaseHTTPRequestHandler):
    """Serve static frontend assets and the read-only evaluation API."""

    server_version = "SentinelAI-Dashboard/2.0"

    def do_GET(self) -> None:
        path = urlparse(self.path).path

        if path == "/api/evaluation":
            self.send_json(build_payload())
            return

        if path == "/api/health":
            self.send_json({"ok": True})
            return

        if path == "/favicon.ico":
            self.send_response(HTTPStatus.NO_CONTENT)
            self.end_headers()
            return

        self.send_static(path)

    def send_json(self, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, separators=(",", ":"), default=str).encode("utf-8")
        self.send_response(HTTPStatus.OK)
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
