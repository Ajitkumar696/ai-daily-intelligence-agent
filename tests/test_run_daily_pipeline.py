import json
import os
import subprocess
from pathlib import Path

import run_daily_pipeline as pipeline


TODAY = "2026-11-07"


def _programme(day=TODAY, status="SUCCESS", narration="A clear daily briefing."):
    return {
        "programme_status": status,
        "programme_date": day,
        "overall_headline": "AI news today",
        "stories": [{"id": "story-1"}],
        "programmes": {
            "one_minute": {"narration": narration},
            "three_minute": {"narration": narration},
        },
    }


def _save_programme(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def _setup(tmp_path, monkeypatch):
    root = tmp_path
    output = root / "output"
    monkeypatch.setattr(pipeline, "ROOT", root)
    monkeypatch.setattr(pipeline, "OUTPUT", output)
    monkeypatch.setattr(pipeline, "PROGRAMME", output / "daily_programme.json")
    monkeypatch.setattr(pipeline, "REPORTS", output / "reports")
    monkeypatch.setattr(pipeline, "VIDEOS", output / "videos")
    monkeypatch.setattr(pipeline, "LOGS", output / "pipeline_logs")
    monkeypatch.setattr(pipeline, "LOCK", output / ".pipeline.lock")
    monkeypatch.setattr(pipeline, "_today", lambda: TODAY)
    return root


def _fake_commands(monkeypatch, *, write_research=True, write_videos=True, commands=None):
    seen = commands if commands is not None else []

    def fake_run(command, **kwargs):
        seen.append(command)
        if "src.run_daily_programme" in command and write_research:
            previous_mtime = pipeline.PROGRAMME.stat().st_mtime_ns if pipeline.PROGRAMME.exists() else 0
            _save_programme(pipeline.PROGRAMME, _programme(narration="  Daily\n  briefing.  "))
            os.utime(pipeline.PROGRAMME, ns=(previous_mtime + 2_000_000_000,
                                             previous_mtime + 2_000_000_000))
            pipeline.REPORTS.mkdir(parents=True, exist_ok=True)
            (pipeline.REPORTS / f"AI_Daily_Intelligence_{TODAY}.pdf").write_bytes(b"pdf")
        if "src.generate_video" in command and write_videos:
            video_type = command[command.index("--type") + 1]
            target = pipeline._video_path(TODAY, video_type)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"mp4")
        return subprocess.CompletedProcess(command, 0, "step output", "")

    monkeypatch.setattr(pipeline.subprocess, "run", fake_run)
    return seen


def test_full_success_runs_research_and_both_videos(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    commands = _fake_commands(monkeypatch)
    result = pipeline.run_pipeline()

    assert result["ok"] is True
    assert result["stage"] == "complete"
    assert result["stories"] == 1
    assert Path(result["pdf"]).is_file()
    assert set(result["videos"]) == {"one_minute", "three_minute"}
    assert len(commands) == 3
    assert not pipeline.LOCK.exists()


def test_idempotent_skip_reuses_today_research_and_runs_videos(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    _save_programme(pipeline.PROGRAMME, _programme())
    pipeline.REPORTS.mkdir(parents=True)
    (pipeline.REPORTS / f"AI_Daily_Intelligence_{TODAY}.pdf").write_bytes(b"pdf")
    commands = _fake_commands(monkeypatch)

    result = pipeline.run_pipeline()

    assert result["ok"] is True
    assert result["skipped_research"] is True
    assert len(commands) == 2
    assert all("src.run_daily_programme" not in command for command in commands)


def test_force_reruns_research_even_for_todays_success(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    _save_programme(pipeline.PROGRAMME, _programme())
    pipeline.REPORTS.mkdir(parents=True)
    (pipeline.REPORTS / f"AI_Daily_Intelligence_{TODAY}.pdf").write_bytes(b"pdf")
    commands = _fake_commands(monkeypatch)

    result = pipeline.run_pipeline(force=True, skip_video=True)

    assert result["ok"] is True
    assert result["skipped_research"] is False
    assert len(commands) == 1
    assert "src.run_daily_programme" in commands[0]


def test_research_exit_zero_without_rewritten_programme_is_failure(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    _save_programme(pipeline.PROGRAMME, _programme(day="2026-11-06"))
    monkeypatch.setattr(pipeline.subprocess, "run", lambda command, **kwargs:
                        subprocess.CompletedProcess(command, 0, "reported failure", ""))

    result = pipeline.run_pipeline(skip_video=True)

    assert result["ok"] is False
    assert result["stage"] == "research"
    assert "not rewritten" in result["error"]


def test_research_nonzero_exit_is_reported_before_programme_validation(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    _save_programme(pipeline.PROGRAMME, _programme(day="2026-11-06"))
    monkeypatch.setattr(
        pipeline.subprocess, "run",
        lambda command, **kwargs: subprocess.CompletedProcess(
            command, 1, "", "No new articles were collected; no daily programme or PDF was created."
        ),
    )

    result = pipeline.run_pipeline(skip_video=True)

    assert result["ok"] is False
    assert result["stage"] == "research"
    assert "daily programme command exited 1" in result["error"]
    assert "No new articles were collected" in result["error"]
    assert "not rewritten" not in result["error"]


def test_stale_programme_is_not_a_success(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    _save_programme(pipeline.PROGRAMME, _programme(day="2026-11-06"))
    _fake_commands(monkeypatch, write_research=False)

    result = pipeline.run_pipeline(skip_video=True)

    assert result["ok"] is False
    assert result["stage"] == "research"


def test_live_lock_returns_already_in_progress(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    pipeline.OUTPUT.mkdir(parents=True)
    pipeline.LOCK.write_text("other process", encoding="utf-8")

    result = pipeline.run_pipeline()

    assert result["ok"] is False
    assert result["error"] == "already in progress"
    assert pipeline.LOCK.exists()


def test_stale_lock_is_removed_and_pipeline_runs(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    pipeline.OUTPUT.mkdir(parents=True)
    pipeline.LOCK.write_text("stale process", encoding="utf-8")
    old = pipeline.time.time() - pipeline.LOCK_MAX_AGE_SECONDS - 10
    os.utime(pipeline.LOCK, (old, old))
    _fake_commands(monkeypatch)

    result = pipeline.run_pipeline(skip_video=True)

    assert result["ok"] is True
    assert not pipeline.LOCK.exists()


def test_missing_mp4_is_reported_as_video_error(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    _save_programme(pipeline.PROGRAMME, _programme())
    pipeline.REPORTS.mkdir(parents=True)
    (pipeline.REPORTS / f"AI_Daily_Intelligence_{TODAY}.pdf").write_bytes(b"pdf")
    _fake_commands(monkeypatch, write_videos=False)

    result = pipeline.run_pipeline(only_type="one_minute")

    assert result["ok"] is False
    assert result["stage"] == "video"
    assert "expected MP4 was not created" in result["error"]


def test_narration_whitespace_is_normalised_only_when_needed(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    _save_programme(pipeline.PROGRAMME, _programme(narration="  One\n  two  "))

    assert pipeline._normalise_narration() is True
    saved = json.loads(pipeline.PROGRAMME.read_text(encoding="utf-8"))
    assert saved["programmes"]["one_minute"]["narration"] == "One two"
    assert saved["programmes"]["three_minute"]["narration"] == "One two"
    assert pipeline._normalise_narration() is False


def test_stdout_is_exactly_one_json_line(tmp_path, monkeypatch, capsys):
    _setup(tmp_path, monkeypatch)
    _fake_commands(monkeypatch)

    assert pipeline.main(["--skip-video"]) == 0
    output = capsys.readouterr().out
    assert output.count("\n") == 1
    assert json.loads(output)["ok"] is True
