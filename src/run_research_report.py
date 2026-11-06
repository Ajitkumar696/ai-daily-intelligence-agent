import argparse
from datetime import date
from pathlib import Path

from src.generate_research import (
    DEFAULT_DAILY_OUTPUT_FILE,
    run_research
)
from src.reporting.pdf_generator import generate_pdf_report
from src.summarization.gemini_client import GeminiClientError
from src.summarization.summarizer import DEFAULT_INPUT_FILE


def run_research_report(
    input_file=None,
    brief_file=None,
    report_dir=None,
    max_topics=None,
    client=None,
    cache=None
):
    """Generate one combined daily programme, persist it, and render one PDF."""

    limit = max_topics if max_topics is not None else 0
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
        raise ValueError("max_topics must be zero or a positive integer.")
    groups, programme, programme_path = run_research(
        input_file=input_file or DEFAULT_INPUT_FILE,
        output_file=brief_file or DEFAULT_DAILY_OUTPUT_FILE,
        max_topics=limit,
        client=client,
        cache=cache
    )
    try:
        pdf_path = generate_pdf_report(
            programme,
            output_dir=report_dir,
            report_date=date.fromisoformat(programme["programme_date"])
        )
    except Exception as error:
        print(
            "Daily programme execution incomplete: PDF generation failed "
            f"({type(error).__name__}). Programme JSON was preserved at "
            f"{programme_path}."
        )
        raise

    print(
        f"Daily AI news programme complete: groups={len(groups)}, "
        f"researched_stories={len(programme['stories'])}, "
        "combined_programmes=2."
    )
    print("Daily programme JSON:", programme_path)
    print("PDF report:", pdf_path)
    return programme, programme_path, pdf_path


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Generate one combined daily AI news package with 1-minute and "
            "3-minute scripts and one PDF report."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_FILE,
        help="Collected article JSON file."
    )
    parser.add_argument(
        "--brief-output",
        type=Path,
        default=DEFAULT_DAILY_OUTPUT_FILE,
        help="Combined daily programme JSON output."
    )
    parser.add_argument("--report-dir", type=Path, default=None)
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help=(
            "Maximum grouped topics sent to the AI provider; "
            "0 includes all groups."
        )
    )
    args = parser.parse_args(argv)
    if args.limit < 0:
        parser.error("--limit must be zero or a positive integer.")
    try:
        return run_research_report(
            input_file=args.input,
            brief_file=args.brief_output,
            report_dir=args.report_dir,
            max_topics=args.limit
        )
    except GeminiClientError as error:
        parser.exit(
            2,
            "AI research failed; no daily programme or PDF was "
            f"generated: {error}\n"
        )


if __name__ == "__main__":
    main()
