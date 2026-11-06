import json

import pytest

import src.run_daily_report as run_daily_report


def test_orchestrator_preserves_json_if_pdf_generation_fails(
    tmp_path,
    monkeypatch
):
    input_path = tmp_path / "articles.json"
    output_path = tmp_path / "summarized_updates.json"
    input_path.write_text(
        json.dumps([{"source": "OpenAI"}]),
        encoding="utf-8"
    )
    saved_records = [
        {
            "title": "Article",
            "source": "OpenAI",
            "source_url": "https://example.com/article",
            "published_date": "",
            "category": "Other",
            "short_summary": "A supported summary.",
            "key_points": ["Supported point."],
            "why_it_matters": "Not specified in the source.",
            "importance": "Low",
            "confidence": 0.5,
            "summary_status": "SUCCESS"
        }
    ]
    monkeypatch.setattr(
        run_daily_report,
        "summarize_articles",
        lambda articles, max_articles: saved_records
    )

    def fail_pdf(*args, **kwargs):
        raise OSError("PDF library unavailable")

    monkeypatch.setattr(run_daily_report, "generate_pdf_report", fail_pdf)

    with pytest.raises(OSError, match="PDF library unavailable"):
        run_daily_report.run_daily_report(
            input_file=input_path,
            summarized_file=output_path,
            report_dir=tmp_path / "reports",
            max_articles=3
        )

    assert json.loads(output_path.read_text(encoding="utf-8")) == saved_records


def test_orchestrator_does_not_run_collection_or_history(tmp_path, monkeypatch):
    input_path = tmp_path / "articles.json"
    input_path.write_text("[]", encoding="utf-8")
    monkeypatch.setattr(
        run_daily_report,
        "summarize_articles",
        lambda articles, max_articles: []
    )
    monkeypatch.setattr(
        run_daily_report,
        "generate_pdf_report",
        lambda summaries, output_dir: tmp_path / "report.pdf"
    )

    summaries, summarized_path, pdf_path = run_daily_report.run_daily_report(
        input_file=input_path,
        summarized_file=tmp_path / "summaries.json",
        report_dir=tmp_path,
        max_articles=0
    )

    assert summaries == []
    assert summarized_path.exists()
    assert pdf_path == tmp_path / "report.pdf"
