import json
from datetime import date
import re

import pytest

import src.generate_report as generate_report
from src.reporting.pdf_generator import (
    NoSuccessfulSummariesError,
    generate_pdf_report,
    next_report_path
)


REPORT_DATE = date(2026, 11, 2)


def valid_summary(**updates):
    article = {
        "title": "A verified AI development",
        "source": "OpenAI",
        "source_url": "https://example.com/article?ref=news",
        "published_date": "2026-11-01",
        "category": "AI Models",
        "short_summary": "The source describes a documented AI development.",
        "key_points": ["The source identifies the described development."],
        "why_it_matters": "The source explains its intended impact.",
        "importance": "High",
        "confidence": 0.9,
        "summary_status": "SUCCESS"
    }
    article.update(updates)
    return article


def valid_research_brief(**updates):
    source_url = "https://example.com/research"
    claim = {
        "statement": "The original source reports this development.",
        "source_urls": [source_url]
    }
    brief = {
        "research_status": "SUCCESS",
        "research_scope": "single_source",
        "source_count": 1,
        "source_urls": [source_url],
        "sources": [
            {
                "publisher": "Example Research",
                "title": "Original research report",
                "url": source_url,
                "canonical_url": source_url,
                "published": "2026-11-02",
                "author": "Example Author",
                "content_source": "full_article"
            }
        ],
        "content_quality": {
            "label": "full_article_content",
            "sources_with_full_articles": 1,
            "sources_using_fallback": 0,
            "sources_without_content": 0
        },
        "cross_check": {
            "metadata_differences": [],
            "claim_agreement": "Not assessed; source-grounded analysis is pending."
        },
        "research_brief": {
            "topic": "AI research",
            "headline": "A verified AI research development",
            "background": [claim],
            "what_happened": [claim],
            "technology_explanation": [claim],
            "important_developments": [
                dict(claim, statement=f"Development {index} is reported.")
                for index in range(1, 4)
            ],
            "technical_details": [claim],
            "companies_models_technologies": [claim],
            "real_world_applications": [claim],
            "benefits": [claim],
            "limitations": [claim],
            "potential_impact": [claim],
            "why_it_matters": [claim],
            "cross_source_comparison": {
                "agreements": [],
                "differences": [],
                "missing_information": [{
                    "detail": "The original source does not specify the test date.",
                    "source_urls": [source_url]
                }]
            },
            "ai_generated_analysis": "This is clearly marked AI analysis.",
            "confidence": 0.87
        },
        "video_script": {
            "suggested_title": "The research explained",
            "thumbnail_text": "AI BREAKTHROUGH?",
            "target_duration_seconds": 60,
            "opening_hook": "Here is the research result.",
            "introduction": "We will explain the source.",
            "main_explanation": "The source describes a development.",
            "important_developments": [
                dict(claim, statement=f"Development {index} is reported.")
                for index in range(1, 4)
            ],
            "practical_examples": [claim],
            "why_viewers_should_care": "The result may be relevant.",
            "closing": "Read the linked source for more.",
            "source_urls": [source_url],
            "scenes": [
                {
                    "scene_number": index,
                    "duration_seconds": 15,
                    "narration": f"Scene {index} explains the finding.",
                    "visual_suggestion": f"Show a diagram for scene {index}.",
                    "on_screen_text": f"Finding {index}",
                    "source_urls": [source_url]
                }
                for index in range(1, 5)
            ]
        }
    }
    brief.update(updates)
    return brief


def require_reportlab():
    try:
        import reportlab
    except ModuleNotFoundError:
        pytest.fail(
            "PDF tests require ReportLab. Install it with "
            "`python -m pip install reportlab==5.1.0`.",
            pytrace=False
        )
    return reportlab


def test_report_edition_numbering_preserves_existing_reports(tmp_path):
    first, first_edition = next_report_path(tmp_path, REPORT_DATE)
    assert first.name == "AI_Daily_Intelligence_2026-11-02.pdf"
    assert first_edition == 1

    first.touch()
    second, second_edition = next_report_path(tmp_path, REPORT_DATE)
    assert second.name == "AI_Daily_Intelligence_2026-11-02_Edition_02.pdf"
    assert second_edition == 2

    second.touch()
    third, third_edition = next_report_path(tmp_path, REPORT_DATE)
    assert third.name == "AI_Daily_Intelligence_2026-11-02_Edition_03.pdf"
    assert third_edition == 3


def test_pdf_generation_has_expected_filename_and_clickable_source(tmp_path):
    require_reportlab()
    output = generate_pdf_report(
        [valid_summary()],
        output_dir=tmp_path,
        report_date=REPORT_DATE
    )

    data = output.read_bytes()
    assert output.name == "AI_Daily_Intelligence_2026-11-02.pdf"
    assert data.startswith(b"%PDF")
    assert data.rstrip().endswith(b"%%EOF")
    assert b"/Type /Catalog" in data
    assert re.search(rb"\bxref\b", data)
    assert b"AI Daily Intelligence" in data
    assert b"Executive summary" in data
    assert b"A verified AI development" in data
    assert b"The source describes a documented AI development." in data
    assert b"Key points" in data
    assert b"Importance: High" in data
    assert b"/URI (https://example.com/article?ref=news)" in data
    assert b"Page 1" in data
    assert b"AI DAILY INTELLIGENCE" in data
    assert len(re.findall(rb"/Type\s*/Page\b", data)) >= 2


def test_research_pdf_includes_verified_briefs_scripts_and_clickable_sources(
    tmp_path
):
    require_reportlab()
    valid = valid_research_brief()
    failed = valid_research_brief(
        research_status="FAILED",
        research_brief={
            "headline": "Failed research must not appear"
        }
    )
    malformed = valid_research_brief(
        research_brief={
            **valid_research_brief()["research_brief"],
            "what_happened": [{
                "statement": "Unsupported",
                "source_urls": ["https://unknown.example/fact"]
            }]
        }
    )

    output = generate_pdf_report(
        [valid, failed, malformed],
        output_dir=tmp_path,
        report_date=REPORT_DATE
    )
    data = output.read_bytes()

    assert data.startswith(b"%PDF")
    assert b"Daily AI news overview" in data
    assert b"Research briefs" in data
    assert b"(1)" in data
    assert b"A verified AI research development" in data
    assert b"Technical details" in data
    assert b"Real-world applications" in data
    assert b"AI-generated analysis" in data
    assert b"Information not covered in supplied content" in data
    assert b"does not specify the test date" in data
    assert b"The research explained" in data
    assert b"Scene 1" in data
    assert b"2026-11-02" in data
    assert b"/URI (https://example.com/research)" in data
    assert b"Failed research must not appear" not in data
    assert b"Unsupported" not in data
    assert b"Page 1" in data


def test_no_successful_research_brief_does_not_create_empty_pdf(tmp_path):
    require_reportlab()
    report_dir = tmp_path / "reports"

    with pytest.raises(NoSuccessfulSummariesError, match="no PDF report"):
        generate_pdf_report(
            [
                valid_research_brief(research_status="FAILED"),
                valid_research_brief(
                    research_brief={
                        "headline": "Invalid research",
                        "what_happened": []
                    }
                )
            ],
            output_dir=report_dir,
            report_date=REPORT_DATE
        )

    assert not report_dir.exists()


@pytest.mark.parametrize(
    "summaries",
    [
        [],
        [
            valid_summary(title="Failed article", summary_status="FAILED"),
            valid_summary(
                title="Insufficient article",
                summary_status="INSUFFICIENT_CONTENT"
            )
        ],
        [
            valid_summary(
                title="Malformed successful summary",
                short_summary="",
                summary_status="SUCCESS"
            )
        ]
    ],
    ids=["empty", "no-successful-records", "invalid-success-record"]
)
def test_zero_valid_summaries_do_not_create_empty_report(
    tmp_path,
    summaries
):
    require_reportlab()
    report_dir = tmp_path / "reports"

    with pytest.raises(NoSuccessfulSummariesError, match="no PDF report"):
        generate_pdf_report(
            summaries,
            output_dir=report_dir,
            report_date=REPORT_DATE
        )

    assert not report_dir.exists()


def test_failed_summaries_are_not_rendered_as_verified_news(tmp_path):
    require_reportlab()
    valid = valid_summary(title="Only verified article")
    failed = valid_summary(
        title="Failed result must not appear",
        summary_status="FAILED"
    )
    output = generate_pdf_report(
        [valid, failed],
        output_dir=tmp_path,
        report_date=REPORT_DATE
    )

    assert output.exists()
    assert output.read_bytes().startswith(b"%PDF")
    assert b"Verified summaries" in output.read_bytes()
    assert b"(1)" in output.read_bytes()
    assert b"Executive summary \\(1 verified article\\)" in output.read_bytes()
    assert b"Only verified article" in output.read_bytes()
    assert b"Failed result must not appear" not in output.read_bytes()


def test_one_valid_summary_mixed_with_failed_and_insufficient_is_included(
    tmp_path
):
    require_reportlab()
    valid = valid_summary(title="Single verified article")
    summaries = [
        valid,
        valid_summary(
            title="Failed article excluded",
            summary_status="FAILED"
        ),
        valid_summary(
            title="Insufficient article excluded",
            summary_status="INSUFFICIENT_CONTENT"
        )
    ]

    output = generate_pdf_report(
        summaries,
        output_dir=tmp_path,
        report_date=REPORT_DATE
    )
    data = output.read_bytes()

    assert b"(3)" in data
    assert b"(1)" in data
    assert b"Executive summary \\(1 verified article\\)" in data
    assert b"1 validated summary is included" in data
    assert b"Single verified article" in data
    assert b"Failed article excluded" not in data
    assert b"Insufficient article excluded" not in data


def test_multiple_valid_summaries_mixed_with_failures_count_exactly(
    tmp_path
):
    require_reportlab()
    summaries = [
        valid_summary(
            title="Verified model article",
            source_url="https://example.com/model"
        ),
        valid_summary(
            title="Verified research article",
            category="Research",
            source_url="https://example.com/research"
        ),
        valid_summary(
            title="Failed article excluded",
            summary_status="FAILED",
            source_url="https://example.com/failed"
        ),
        valid_summary(
            title="Malformed success excluded",
            category="Invalid category",
            source_url="https://example.com/malformed"
        )
    ]

    output = generate_pdf_report(
        summaries,
        output_dir=tmp_path,
        report_date=REPORT_DATE
    )
    data = output.read_bytes()

    assert b"Articles processed" in data
    assert b"(4)" in data
    assert b"Verified summaries" in data
    assert b"(2)" in data
    assert b"Executive summary \\(2 verified articles\\)" in data
    assert b"2 validated summaries are included" in data
    assert b"Verified model article" in data
    assert b"Verified research article" in data
    assert b"Failed article excluded" not in data
    assert b"Malformed success excluded" not in data


def test_multiple_categories_create_sectioned_multipage_report(tmp_path):
    require_reportlab()
    summaries = [
        valid_summary(category="AI Models"),
        valid_summary(
            title="A separate research finding",
            category="Research",
            source_url="https://example.com/research"
        )
    ]
    output = generate_pdf_report(
        summaries,
        output_dir=tmp_path,
        report_date=REPORT_DATE
    )
    data = output.read_bytes()

    assert data.startswith(b"%PDF")
    assert len(re.findall(rb"/Type\s*/Page\b", data)) >= 2
    assert b"/URI (https://example.com/research)" in data


def test_pdf_tolerates_missing_optional_article_fields(tmp_path):
    require_reportlab()
    article = valid_summary()
    article.pop("published_date")
    article.pop("source")

    output = generate_pdf_report(
        [article],
        output_dir=tmp_path,
        report_date=REPORT_DATE
    )

    assert output.exists()


def test_long_article_content_flows_across_pages(tmp_path):
    require_reportlab()
    long_text = (
        "The source carefully explains a documented research result and "
        "describes the conditions and limits of its evaluation. "
    ) * 150
    article = valid_summary(
        short_summary=long_text[:650],
        key_points=[long_text for _ in range(5)]
    )

    output = generate_pdf_report(
        [article] * 4,
        output_dir=tmp_path,
        report_date=REPORT_DATE
    )
    data = output.read_bytes()

    assert output.exists()
    assert len(re.findall(rb"/Type\s*/Page\b", data)) > 2


def test_long_source_url_wraps_and_remains_clickable(tmp_path):
    require_reportlab()
    url = "https://example.com/" + ("very-long-path-segment-" * 30)
    output = generate_pdf_report(
        [valid_summary(source_url=url)],
        output_dir=tmp_path,
        report_date=REPORT_DATE
    )

    data = output.read_bytes()
    assert output.exists()
    assert f"/URI ({url})".encode() in data
    assert b"Open original source:" in data


def test_each_pdf_generation_keeps_previous_edition(tmp_path):
    require_reportlab()
    first = generate_pdf_report(
        [valid_summary()],
        output_dir=tmp_path,
        report_date=REPORT_DATE
    )
    first_content = first.read_bytes()
    second = generate_pdf_report(
        [valid_summary()],
        output_dir=tmp_path,
        report_date=REPORT_DATE
    )

    assert first != second
    assert first.exists()
    assert second.exists()
    assert first.read_bytes() == first_content
    assert second.name.endswith("_Edition_02.pdf")


def test_generate_report_reads_the_default_summarized_json(tmp_path, monkeypatch):
    input_path = tmp_path / "summarized_updates.json"
    summaries = [valid_summary()]
    input_path.write_text(
        json.dumps(summaries),
        encoding="utf-8"
    )
    output_path = tmp_path / "report.pdf"
    received = {}

    def fake_generate_pdf_report(items, output_dir=None):
        received["items"] = items
        received["output_dir"] = output_dir
        return output_path

    monkeypatch.setattr(generate_report, "DEFAULT_OUTPUT_FILE", input_path)
    monkeypatch.setattr(
        generate_report,
        "generate_pdf_report",
        fake_generate_pdf_report
    )

    assert generate_report.main([]) == output_path
    assert received == {"items": summaries, "output_dir": None}


@pytest.mark.parametrize(
    ("file_content", "error"),
    [
        ("not JSON", json.JSONDecodeError),
        ('{"not": "a list"}', SystemExit)
    ]
)
def test_generate_report_rejects_invalid_input_json(
    tmp_path,
    file_content,
    error
):
    input_path = tmp_path / "invalid.json"
    input_path.write_text(file_content, encoding="utf-8")

    with pytest.raises(error):
        generate_report.main(["--input", str(input_path)])


def test_generate_report_reports_missing_input_file(tmp_path):
    missing_path = tmp_path / "missing.json"

    with pytest.raises(FileNotFoundError):
        generate_report.main(["--input", str(missing_path)])
