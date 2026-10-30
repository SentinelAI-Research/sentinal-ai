import json
import time
from http.server import ThreadingHTTPServer
from threading import Thread
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from backend import dashboard_server
from backend.dashboard_server import DashboardHandler


@pytest.fixture
def dashboard_url():
    with dashboard_server.AGENT_RUNS_LOCK:
        dashboard_server.AGENT_RUNS.clear()
    server = ThreadingHTTPServer(("127.0.0.1", 0), DashboardHandler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()
    thread.join()
    server.server_close()
    with dashboard_server.AGENT_RUNS_LOCK:
        dashboard_server.AGENT_RUNS.clear()


def post_scenario(base_url, scenario_id):
    request = Request(
        f"{base_url}/api/agent/runs",
        data=json.dumps({"scenario_id": scenario_id}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request) as response:
        return response.status, json.loads(response.read())


def test_catalog_and_safe_run_endpoint_return_actual_trace(dashboard_url):
    with urlopen(f"{dashboard_url}/api/agent/scenarios") as response:
        catalog = json.loads(response.read())
    assert len(catalog["scenarios"]) == 2

    status, result = post_scenario(dashboard_url, "safe-summary")
    assert status == 201
    assert result["status"] == "completed"
    assert result["summary"]["executed_actions"] == 3
    assert [step["tool_name"] for step in result["trace"]] == [
        "read_file",
        "summarize",
        "paraphrase",
    ]

    with urlopen(f"{dashboard_url}/api/agent/runs/{result['run_id']}") as response:
        fetched = json.loads(response.read())
    assert fetched == result


def test_blocked_scenario_returns_denial_without_execution(dashboard_url):
    _, result = post_scenario(
        dashboard_url,
        "confidential-external-message",
    )

    assert result["status"] == "stopped"
    assert result["summary"]["blocked_actions"] == 1
    assert result["trace"][0]["verdict"] == "BLOCK"
    assert result["trace"][0]["execution_status"] == "not_attempted"
    assert result["trace"][0]["output_data_object_ids"] == []


def test_api_rejects_unknown_scenario_and_arbitrary_plan(dashboard_url):
    for payload in (
        {"scenario_id": "not-a-real-scenario"},
        {"scenario_id": "safe-summary", "actions": [{"tool_name": "send_message"}]},
    ):
        request = Request(
            f"{dashboard_url}/api/agent/runs",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with pytest.raises(HTTPError) as error:
            urlopen(request)
        assert error.value.code in {400, 404}


def test_api_rejects_invalid_json_and_unknown_run(dashboard_url):
    request = Request(
        f"{dashboard_url}/api/agent/runs",
        data=b"not-json",
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with pytest.raises(HTTPError) as error:
        urlopen(request)
    assert error.value.code == 400

    with pytest.raises(HTTPError) as error:
        urlopen(f"{dashboard_url}/api/agent/runs/missing")
    assert error.value.code == 404


def test_llm_configuration_reports_availability_without_secrets(
    dashboard_url, monkeypatch
):
    secret = "test-secret-must-not-be-returned"
    monkeypatch.setenv("OPENAI_API_KEY", secret)
    monkeypatch.setenv("SENTINELAI_OPENAI_MODEL", "test-model")

    with urlopen(f"{dashboard_url}/api/llm/configuration") as response:
        payload = json.loads(response.read())

    serialized = json.dumps(payload)
    assert payload["openai"]["configured"] is True
    assert payload["openai"]["model"] == "test-model"
    assert secret not in serialized
    assert "call_external_api" not in serialized
    assert payload["cancellation_supported"] is False


def test_llm_fake_run_api_returns_events_and_provenance(dashboard_url):
    request = Request(
        f"{dashboard_url}/api/llm/runs",
        data=json.dumps(
            {
                "task": "Calculate six times seven using the calculator fixture.",
                "mode": "fake",
                "scenario_id": "calculator",
                "policy_id": "default",
                "max_actions": 4,
            }
        ).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request) as response:
        assert response.status == 202
        accepted = json.loads(response.read())
    run_id = accepted["run_id"]

    deadline = time.monotonic() + 8
    while time.monotonic() < deadline:
        with urlopen(f"{dashboard_url}/api/llm/runs/{run_id}") as response:
            run = json.loads(response.read())
        if run["status"] in {"completed", "stopped", "failed"}:
            break
        time.sleep(0.01)

    assert run["status"] == "completed"
    assert run["mode"] == "fake"
    assert run["summary"]["executed"] == 1
    assert run["events"]

    with urlopen(f"{dashboard_url}/api/llm/runs/{run_id}/events") as response:
        events = json.loads(response.read())["events"]
    assert any(event["type"] == "monitor_decision" for event in events)
    assert any(event["type"] == "output_provenance_recorded" for event in events)

    with urlopen(f"{dashboard_url}/api/llm/runs/{run_id}/provenance") as response:
        provenance = json.loads(response.read())
    assert provenance["objects"]
    assert provenance["edges"]

    with urlopen(f"{dashboard_url}/api/llm/runs") as response:
        history = json.loads(response.read())["runs"]
    assert history[0]["run_id"] == run_id
    assert history[0]["summary"]["executed"] == 1


def test_live_run_is_rejected_if_sdk_or_credentials_are_unavailable(
    dashboard_url, monkeypatch
):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    request = Request(
        f"{dashboard_url}/api/llm/runs",
        data=json.dumps(
            {
                "task": "Run a safe task.",
                "mode": "live",
                "scenario_id": "safe_document",
                "policy_id": "default",
                "max_actions": 3,
            }
        ).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with pytest.raises(HTTPError) as error:
        urlopen(request)
    assert error.value.code == 503
    assert json.loads(error.value.read())["error"] in {
        "provider_not_configured",
        "provider_sdk_unavailable",
    }
