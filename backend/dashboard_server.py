"""Small data-backed HTTP server for the SentinelAI research dashboard.

The dashboard intentionally reads evaluation output at request time. It never
loads the trained joblib artifacts and never embeds model scores in frontend
code.
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
RESULT_PATH = ROOT / "results" / "ml_evaluation" / (
    "provenance_hard_negative_evaluation.json"
)
MODEL_DIR = ROOT / "models" / "hard_negative"
MODEL_METADATA = {
    "without_provenance": MODEL_DIR / (
        "security_risk_hard_negative_without_provenance_metadata.json"
    ),
    "with_provenance": MODEL_DIR / (
        "security_risk_hard_negative_with_provenance_metadata.json"
    ),
}
DATA_DIR = ROOT / "data" / "processed" / "hard_negative"

LABELS = ["SAFE", "EARLY_RISK", "VIOLATION"]
PROMPT_EVALUATION_FALLBACK = {
    "without_provenance": {
        "overall": {
            "accuracy": 0.9062,
            "macro_precision": 0.5729,
            "macro_recall": 0.6667,
            "macro_f1": 0.6121,
            "weighted_f1": 0.8670,
        },
        "final_step": {"accuracy": 0.4375},
        "final_safe_vs_violation": {"accuracy": 0.4375},
    },
    "with_provenance": {
        "overall": {
            "accuracy": 1.0,
            "macro_precision": 1.0,
            "macro_recall": 1.0,
            "macro_f1": 1.0,
            "weighted_f1": 1.0,
        },
        "final_step": {"accuracy": 1.0},
        "final_safe_vs_violation": {"accuracy": 1.0},
    },
    "improvement": {
        "accuracy": 0.0938,
        "macro_f1": 0.3879,
        "final_step_accuracy": 0.5625,
        "final_safe_vs_violation_accuracy": 0.5625,
    },
}
EXPERIMENT_DESIGN = {
    "source_note": "Documented experiment design",
    "ami_trajectories": 100,
    "enron_trajectories": 100,
    "steps_per_trajectory": 6,
    "ami_records": 600,
    "enron_records": 600,
    "total_records": 1200,
    "total_trajectories": 200,
}


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def iso_mtime(path: Path) -> str | None:
    if not path.exists():
        return None
    return datetime.fromtimestamp(
        path.stat().st_mtime, tz=timezone.utc
    ).isoformat()


def dataset_snapshot() -> dict[str, Any]:
    """Read available JSONL records without requiring the ML toolchain."""
    paths = sorted(DATA_DIR.glob("*.jsonl"))
    if not paths:
        return {
            "available": False,
            "record_count": None,
            "trajectory_count": None,
            "source_datasets": {},
            "trajectory": [],
        }

    record_count = 0
    trajectory_ids: set[str] = set()
    source_counts: dict[str, int] = {}
    first_trajectory_id: str | None = None
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
                    record_count += 1
                    trajectory_id = str(record.get("trajectory_id", ""))
                    if trajectory_id:
                        trajectory_ids.add(trajectory_id)
                    source = str(record.get("source_dataset", "unknown"))
                    source_counts[source] = source_counts.get(source, 0) + 1
                    if first_trajectory_id is None and trajectory_id:
                        first_trajectory_id = trajectory_id
                    if (
                        trajectory_id
                        and trajectory_id == first_trajectory_id
                    ):
                        first_trajectory.append(
                            {
                                "step": record.get("step_number"),
                                "tool": record.get("tool_name"),
                                "label": record.get("label"),
                                "sensitivity": record.get("sensitivity"),
                                "has_provenance": record.get(
                                    "has_provenance"
                                ),
                                "provenance_depth": record.get(
                                    "provenance_depth"
                                ),
                                "transformation": record.get(
                                    "transformation"
                                ),
                            }
                        )
        except OSError:
            continue

    first_trajectory.sort(
        key=lambda item: int(item["step"] or 0)
    )
    return {
        "available": record_count > 0,
        "record_count": record_count or None,
        "trajectory_count": len(trajectory_ids) or None,
        "source_datasets": source_counts,
        "trajectory": first_trajectory,
    }


def feature_payload(metadata: dict[str, Any] | None) -> list[str]:
    if not metadata:
        return []
    features = metadata.get("feature_names")
    return features if isinstance(features, list) else []


def build_payload() -> dict[str, Any]:
    evaluation = read_json(RESULT_PATH)
    using_prompt_snapshot = evaluation is None
    if using_prompt_snapshot:
        evaluation = PROMPT_EVALUATION_FALLBACK
    metadata = {
        name: read_json(path) for name, path in MODEL_METADATA.items()
    }
    return {
        "status": (
            "prompt_snapshot" if using_prompt_snapshot else "ready"
        ),
        "evaluation_source": str(
            RESULT_PATH.relative_to(ROOT)
        ),
        "evaluation_is_fallback": using_prompt_snapshot,
        "evaluation_updated_at": iso_mtime(RESULT_PATH),
        "evaluation": evaluation,
        "models": {
            "without_provenance": {
                "name": (
                    "security_risk_hard_negative_without_provenance"
                ),
                "features": feature_payload(
                    metadata["without_provenance"]
                ),
                "metadata_available": metadata["without_provenance"]
                is not None,
            },
            "with_provenance": {
                "name": "security_risk_hard_negative_with_provenance",
                "features": feature_payload(
                    metadata["with_provenance"]
                ),
                "metadata_available": metadata["with_provenance"]
                is not None,
            },
        },
        "dataset": dataset_snapshot(),
        "experiment_design": EXPERIMENT_DESIGN,
        "labels": LABELS,
    }


class DashboardHandler(BaseHTTPRequestHandler):
    """Serve the static dashboard and a small read-only JSON API."""

    server_version = "SentinelAI-Dashboard/1.0"

    def do_GET(self) -> None:  # noqa: N802
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
        body = json.dumps(payload, separators=(",", ":")).encode()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def send_static(self, path: str) -> None:
        relative = "index.html" if path in {"/", ""} else path.lstrip("/")
        candidate = (FRONTEND_DIR / relative).resolve()
        if (
            FRONTEND_DIR.resolve() not in candidate.parents
            or not candidate.is_file()
        ):
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        body = candidate.read_bytes()
        content_type = mimetypes.guess_type(candidate.name)[0]
        self.send_response(HTTPStatus.OK)
        self.send_header(
            "Content-Type", content_type or "application/octet-stream"
        )
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:
        print(f"[dashboard] {format % args}")


def main() -> None:
    port = int(os.environ.get("PORT", "5000"))
    server = ThreadingHTTPServer(("0.0.0.0", port), DashboardHandler)
    print(f"SentinelAI dashboard listening on http://0.0.0.0:{port}")
    server.serve_forever()


if __name__ == "__main__":
    main()