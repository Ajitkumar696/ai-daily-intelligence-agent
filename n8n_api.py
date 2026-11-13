
from __future__ import annotations

import hmac
import json
import logging
import os
import re
import subprocess
import sys
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, status
from dotenv import load_dotenv
from files_api import files_router


PROJECT_ROOT = Path(__file__).resolve().parent
PIPELINE_SCRIPT = PROJECT_ROOT / "run_daily_pipeline.py"
load_dotenv(dotenv_path=PROJECT_ROOT / ".env", override=False)

app = FastAPI(title="AI Daily Intelligence API", docs_url=None, redoc_url=None)

_jobs: dict[str, "Job"] = {}
_jobs_lock = threading.Lock()

_SECRET_PATTERN = re.compile(
    r"(?i)(api[_ -]?key|access[_ -]?token|authorization|password|secret)"
    r"(\s*[:=]\s*)\S+|\bBearer\s+\S+|\b(?:gsk_[A-Za-z0-9_-]{12,}|"
    r"sk-[A-Za-z0-9_-]{12,}|AIza[A-Za-z0-9_-]{20,})\b"
)

_logger = logging.getLogger(__name__)


@dataclass
class Job:
    process: Any
    status: str = "running"
    result: dict[str, Any] | None = None


def _authorize(authorization: str | None) -> None:
    expected = os.environ.get("N8N_API_TOKEN", "")

    if not expected:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="API token is not configured",
        )

    scheme, _, supplied = (authorization or "").partition(" ")

    if (
        scheme.lower() != "bearer"
        or not supplied
        or not hmac.compare_digest(
            supplied.encode("utf-8"),
            expected.encode("utf-8"),
        )
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized",
            headers={"WWW-Authenticate": "Bearer"},
        )


def _sanitize(value: Any) -> Any:
    if isinstance(value, str):
        safe = value

        for name, secret in os.environ.items():
            if (
                any(
                    part in name.upper()
                    for part in ("KEY", "TOKEN", "SECRET", "PASSWORD")
                )
                and len(secret) >= 6
            ):
                safe = safe.replace(secret, "[REDACTED]")

        return _SECRET_PATTERN.sub(
            lambda match: (
                "[REDACTED]"
                if match.lastindex is None
                else match.group(1) + match.group(2) + "[REDACTED]"
            ),
            safe,
        )

    if isinstance(value, list):
        return [_sanitize(item) for item in value]

    if isinstance(value, dict):
        return {
            str(key): _sanitize(item)
            for key, item in value.items()
        }

    return value


def _extract_pipeline_result(
    stdout: str | None,
) -> dict[str, Any] | None:
    """Find the runner's result object among progress or trailing console text."""

    if not stdout:
        return None

    decoder = json.JSONDecoder()
    candidates: list[dict[str, Any]] = []

    required = {
        "ok",
        "date",
        "stage",
        "error",
        "stories",
        "headline",
        "pdf",
        "videos",
        "skipped_research",
        "duration_seconds",
        "log",
    }

    for offset, char in enumerate(stdout):
        if char != "{":
            continue

        try:
            value, _end = decoder.raw_decode(stdout, offset)
        except json.JSONDecodeError:
            continue

        if (
            isinstance(value, dict)
            and required.issubset(value)
            and isinstance(value.get("ok"), bool)
            and isinstance(value.get("stage"), str)
        ):
            candidates.append(value)

    return candidates[-1] if candidates else None


def _monitor_job(job_id: str, job: Job) -> None:
    try:
        stdout, stderr = job.process.communicate()
        return_code = job.process.returncode

        payload = _extract_pipeline_result(stdout)

        parse_state = (
            "result found"
            if payload is not None
            else "empty stdout"
            if not stdout
            else "no valid runner result JSON object found"
        )

        log_method = (
            _logger.info
            if payload is not None and return_code == 0
            else _logger.warning
        )

        log_method(
            "Pipeline job %s exited with return code %s; stdout parsing: %s",
            job_id,
            return_code,
            parse_state,
        )

        if return_code != 0:
            diagnostic = _sanitize(
                (stderr or stdout or "")[-4000:]
            )

            _logger.error(
                "Pipeline job %s failure details:\n%s",
                job_id,
                diagnostic or "[No stdout or stderr captured]",
            )

        if return_code == 0 and payload is not None:
            job.result = _sanitize(payload)
            job.status = (
                "success"
                if job.result.get("ok") is True
                else "failed"
            )
        else:
            job.result = {
                "error": "Pipeline subprocess failed or returned no valid JSON."
            }
            job.status = "failed"

    except Exception as error:
        _logger.error(
            "Pipeline job %s monitoring failed (%s)",
            job_id,
            type(error).__name__,
        )

        job.result = {
            "error": "Pipeline subprocess could not be monitored."
        }
        job.status = "failed"


@app.get("/")
def root() -> dict[str, str]:
    return {
        "status": "ok",
        "service": "AI Daily Intelligence API",
        "health": "/health",
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/run", status_code=status.HTTP_202_ACCEPTED)
def run_pipeline(
    authorization: str | None = Header(default=None),
) -> dict[str, str]:

    _authorize(authorization)

    with _jobs_lock:
        if any(job.status == "running" for job in _jobs.values()):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="A pipeline run is already in progress",
            )

        job_id = str(uuid.uuid4())

        try:
            process = subprocess.Popen(
                [sys.executable, str(PIPELINE_SCRIPT)],
                cwd=str(PROJECT_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                env={
                    **os.environ,
                    "PYTHONIOENCODING": "utf-8",
                    "PYTHONUTF8": "1",
                },
            )

        except (OSError, subprocess.SubprocessError):
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Could not start the pipeline",
            ) from None

        job = Job(process=process)
        _jobs[job_id] = job

        threading.Thread(
            target=_monitor_job,
            args=(job_id, job),
            daemon=True,
        ).start()

    return {
        "job_id": job_id,
        "status": "running",
    }


@app.get("/status/{job_id}")
def job_status(
    job_id: str,
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:

    _authorize(authorization)

    with _jobs_lock:
        job = _jobs.get(job_id)

        if job is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Unknown job ID",
            )

        return {
            "job_id": job_id,
            "status": job.status,
            "result": job.result,
        }


def _require_files_auth(
    authorization: str | None = Header(default=None),
) -> None:
    _authorize(authorization)


app.include_router(
    files_router,
    dependencies=[Depends(_require_files_auth)],
)
