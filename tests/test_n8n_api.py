import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import n8n_api
import files_api


client = TestClient(n8n_api.app)
TOKEN = "test-token-value"


def _runner_result(**updates):
    result = {
        "ok": True, "date": "2026-11-10", "stage": "complete", "error": "",
        "stories": 1, "headline": "Daily briefing", "pdf": "report.pdf",
        "videos": {"one_minute": "one.mp4", "three_minute": "three.mp4"},
        "skipped_research": False, "duration_seconds": 1, "log": "pipeline.log",
    }
    result.update(updates)
    return result


def setup_function():
    n8n_api._jobs.clear()


def test_health_is_public():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_run_and_status_require_token(monkeypatch):
    monkeypatch.setenv("N8N_API_TOKEN", TOKEN)
    assert client.post("/run").status_code == 401
    assert client.get("/status/missing").status_code == 401


def test_duplicate_run_is_rejected_and_job_status(monkeypatch):
    monkeypatch.setenv("N8N_API_TOKEN", TOKEN)
    process = SimpleNamespace(returncode=None, communicate=lambda: ("", ""))
    monkeypatch.setattr(n8n_api.subprocess, "Popen", lambda *args, **kwargs: process)

    class HeldThread:
        def __init__(self, target, args, daemon):
            self.target, self.args = target, args

        def start(self):
            pass

    monkeypatch.setattr(n8n_api.threading, "Thread", HeldThread)
    headers = {"Authorization": f"Bearer {TOKEN}"}
    started = client.post("/run", headers=headers)
    assert started.status_code == 202
    job_id = started.json()["job_id"]
    assert client.post("/run", headers=headers).status_code == 409
    assert client.get(f"/status/{job_id}", headers=headers).json()["status"] == "running"

    process.returncode = 0
    process.communicate = lambda: (json.dumps(_runner_result()), "secret stderr")
    n8n_api._monitor_job(job_id, n8n_api._jobs[job_id])
    status_response = client.get(f"/status/{job_id}", headers=headers)
    assert status_response.status_code == 200
    assert status_response.json() == {
        "job_id": job_id, "status": "success", "result": _runner_result()
    }
    assert client.get("/status/unknown", headers=headers).status_code == 404


def test_root_returns_service_links():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "AI Daily Intelligence API",
        "health": "/health",
    }


@pytest.mark.parametrize("stdout", [
    json.dumps(_runner_result()),
    "collecting sources...\nresearch complete\n" + json.dumps(_runner_result()),
    json.dumps(_runner_result()) + "\nvideo renderer: done\nadditional console output\n",
])
def test_monitor_extracts_runner_json_despite_surrounding_output(stdout):
    job = n8n_api.Job(process=SimpleNamespace(communicate=lambda: (stdout, "sensitive stderr"), returncode=0))

    n8n_api._monitor_job("test-job", job)

    assert job.status == "success"
    assert job.result["ok"] is True
    assert "sensitive stderr" not in json.dumps(job.result)


@pytest.mark.parametrize("stdout", ["not json", "", '{"ok": true,'])
def test_monitor_marks_missing_or_invalid_json_failed(stdout, caplog):
    job = n8n_api.Job(process=SimpleNamespace(communicate=lambda: (stdout, "private stderr"), returncode=0))

    n8n_api._monitor_job("test-job", job)

    assert job.status == "failed"
    assert job.result == {"error": "Pipeline subprocess failed or returned no valid JSON."}
    assert "return code 0" in caplog.text
    assert "private stderr" not in caplog.text


def test_monitor_marks_nonzero_exit_failed_even_with_valid_json():
    job = n8n_api.Job(process=SimpleNamespace(
        communicate=lambda: (json.dumps(_runner_result()), "private stderr"), returncode=2
    ))

    n8n_api._monitor_job("test-job", job)

    assert job.status == "failed"
    assert job.result == {"error": "Pipeline subprocess failed or returned no valid JSON."}


def test_file_routes_require_the_shared_bearer_token(monkeypatch, tmp_path):
    monkeypatch.setenv("N8N_API_TOKEN", TOKEN)
    monkeypatch.setattr(files_api, "OUTPUT", tmp_path / "output")
    monkeypatch.setattr(files_api, "REPORTS", tmp_path / "output" / "reports")
    monkeypatch.setattr(files_api, "VIDEOS", tmp_path / "output" / "videos")
    assert client.get("/files/manifest").status_code == 401
    assert client.get("/files/pdf", headers={"Authorization": f"Bearer {TOKEN}"}).status_code == 404


def test_manifest_and_latest_downloads_are_confined_to_generated_outputs(tmp_path, monkeypatch):
    output = tmp_path / "output"
    reports = output / "reports"
    videos = output / "videos"
    reports.mkdir(parents=True)
    videos.mkdir()
    (output / "daily_programme.json").write_text('{"ok": true}', encoding="utf-8")
    old_pdf = reports / "AI_Daily_Intelligence_2026-11-05.pdf"
    latest_pdf = reports / "AI_Daily_Intelligence_2026-11-06_Edition_02.pdf"
    old_pdf.write_bytes(b"old")
    latest_pdf.write_bytes(b"latest-pdf")
    (reports / "secrets.pdf").write_bytes(b"never-serve")
    one = videos / "AI_Daily_Intelligence_2026-11-06_one_minute"
    three = videos / "AI_Daily_Intelligence_2026-11-06_three_minute"
    one.mkdir()
    three.mkdir()
    (one / "AI_Daily_Intelligence_2026-11-06.mp4").write_bytes(b"one")
    (three / "AI_Daily_Intelligence_2026-11-06.mp4").write_bytes(b"three")
    monkeypatch.setattr(files_api, "OUTPUT", output)
    monkeypatch.setattr(files_api, "REPORTS", reports)
    monkeypatch.setattr(files_api, "VIDEOS", videos)
    monkeypatch.setenv("N8N_API_TOKEN", TOKEN)
    headers = {"Authorization": f"Bearer {TOKEN}"}

    manifest = client.get("/files/manifest", headers=headers)
    assert manifest.status_code == 200
    data = manifest.json()
    assert data["daily_programme"]["filename"] == "daily_programme.json"
    assert data["pdf"]["filename"] == latest_pdf.name
    assert data["pdf"]["size"] == len(b"latest-pdf")
    assert "last_modified" in data["pdf"]
    assert str(tmp_path) not in manifest.text
    assert data["videos"]["one_minute"]["filename"] == "AI_Daily_Intelligence_2026-11-06.mp4"
    assert client.get("/files/pdf", headers=headers).content == b"latest-pdf"
    assert client.get("/files/video/one_minute", headers=headers).content == b"one"
    assert client.get("/files/video/three_minute", headers=headers).content == b"three"


def test_file_routes_return_404_when_outputs_are_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(files_api, "OUTPUT", tmp_path / "output")
    monkeypatch.setattr(files_api, "REPORTS", tmp_path / "output" / "reports")
    monkeypatch.setattr(files_api, "VIDEOS", tmp_path / "output" / "videos")
    monkeypatch.setenv("N8N_API_TOKEN", TOKEN)
    headers = {"Authorization": f"Bearer {TOKEN}"}
    assert client.get("/files/pdf", headers=headers).status_code == 404
    assert client.get("/files/video/one_minute", headers=headers).status_code == 404
    assert client.get("/files/video/three_minute", headers=headers).status_code == 404
