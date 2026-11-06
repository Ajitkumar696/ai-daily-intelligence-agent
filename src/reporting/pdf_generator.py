import os
import re
import tempfile
from datetime import date
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPORT_DIR = PROJECT_ROOT / "output" / "reports"
REPORT_STEM = "AI_Daily_Intelligence"


class NoSuccessfulSummariesError(ValueError):
    """Raised when a report contains no validated successful summaries."""


def next_report_path(report_dir, report_date):
    """Return a non-overwriting date-based path and its edition number."""

    directory = Path(report_dir)
    date_text = report_date.isoformat()
    base = directory / f"{REPORT_STEM}_{date_text}.pdf"
    existing_editions = [1] if base.exists() else []
    pattern = re.compile(
        rf"^{re.escape(REPORT_STEM)}_{re.escape(date_text)}_Edition_(\d+)\.pdf$"
    )

    if directory.exists():
        for candidate in directory.iterdir():
            match = pattern.match(candidate.name)
            if match:
                existing_editions.append(int(match.group(1)))

    if not existing_editions:
        return base, 1

    edition = max(existing_editions) + 1
    return (
        directory / f"{REPORT_STEM}_{date_text}_Edition_{edition:02d}.pdf",
        edition
    )


def _draw_page_chrome(canvas, document):
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.units import inch

    canvas.saveState()
    page_width, page_height = letter
    page_number = canvas.getPageNumber()

    canvas.setStrokeColor(colors.HexColor("#D6E0E7"))
    canvas.setLineWidth(0.5)
    canvas.line(
        document.leftMargin,
        page_height - 0.55 * inch,
        page_width - document.rightMargin,
        page_height - 0.55 * inch
    )
    canvas.setFont("Helvetica-Bold", 8)
    canvas.setFillColor(colors.HexColor("#10243A"))
    canvas.drawString(
        document.leftMargin,
        page_height - 0.42 * inch,
        "AI DAILY INTELLIGENCE"
    )

    canvas.setStrokeColor(colors.HexColor("#D6E0E7"))
    canvas.setLineWidth(0.5)
    canvas.line(
        document.leftMargin,
        0.55 * inch,
        page_width - document.rightMargin,
        0.55 * inch
    )
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#647786"))
    canvas.drawString(
        document.leftMargin,
        0.36 * inch,
        "AI Daily Intelligence • Source-grounded briefing"
    )
    canvas.drawRightString(
        page_width - document.rightMargin,
        0.36 * inch,
        f"Page {page_number}"
    )
    canvas.restoreState()


def generate_pdf_report(
    summaries,
    output_dir=None,
    report_date=None
):
    """Generate an atomic, edition-numbered PDF without overwriting reports."""

    report_date = report_date or date.today()
    if not isinstance(report_date, date):
        raise TypeError("report_date must be a datetime.date instance.")

    daily_programme_mode = (
        isinstance(summaries, dict)
        and "programme_status" in summaries
    )
    research_mode = (
        isinstance(summaries, list)
        and any(
            isinstance(item, dict) and "research_status" in item
            for item in summaries
        )
    )
    if daily_programme_mode:
        from .research_sections import (
            build_daily_programme_story
        )
        from ..research.daily_programme import validate_daily_programme_record

        successful = validate_daily_programme_record(summaries)
        story_builder = build_daily_programme_story
        if successful["programme_date"] != report_date.isoformat():
            raise ValueError(
                "Daily programme date does not match the requested PDF date."
            )
    elif research_mode:
        from .research_sections import (
            build_research_report_story,
            successful_research_briefs
        )

        successful = successful_research_briefs(summaries)
        story_builder = build_research_report_story
    else:
        from .sections import _successful_summaries, build_report_story

        successful = _successful_summaries(summaries)
        story_builder = build_report_story
    if not successful:
        raise NoSuccessfulSummariesError(
            "No valid successful summaries or research briefs are available; "
            "no PDF report was created."
        )

    directory = Path(output_dir) if output_dir is not None else DEFAULT_REPORT_DIR
    directory.mkdir(parents=True, exist_ok=True)
    output_path = None
    temporary_path = None

    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.units import inch
        from reportlab.platypus import SimpleDocTemplate

        while output_path is None:
            candidate, edition = next_report_path(directory, report_date)
            try:
                file_descriptor = os.open(
                    candidate,
                    os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                    0o666
                )
            except FileExistsError:
                continue
            os.close(file_descriptor)
            output_path = candidate

        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=directory,
            prefix=f".{REPORT_STEM}_{report_date.isoformat()}_",
            suffix=".tmp.pdf",
            delete=False
        ) as file:
            temporary_path = Path(file.name)

        document = SimpleDocTemplate(
            str(temporary_path),
            pagesize=letter,
            rightMargin=0.72 * inch,
            leftMargin=0.72 * inch,
            topMargin=0.78 * inch,
            bottomMargin=0.78 * inch,
            pageCompression=0,
            title="AI Daily Intelligence",
            author="AI Daily Intelligence Agent",
            subject="Source-grounded artificial intelligence news briefing"
        )
        story = story_builder(summaries, edition, report_date)
        document.build(
            story,
            onFirstPage=_draw_page_chrome,
            onLaterPages=_draw_page_chrome
        )
        os.replace(temporary_path, output_path)
        temporary_path = None
    except Exception:
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass
        if output_path is not None:
            try:
                output_path.unlink()
            except FileNotFoundError:
                pass
        raise

    return output_path
