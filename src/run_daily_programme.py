import argparse
from datetime import date
from pathlib import Path

from src.collect import run_pipeline
from src.generate_research import DEFAULT_DAILY_OUTPUT_FILE
from src.run_research_report import run_research_report
from src.summarization.summarizer import DEFAULT_INPUT_FILE
from src.summarization.gemini_client import GeminiClientError


def run_daily_programme(
    input_file=None,
    programme_file=None,
    report_dir=None,
    history_file=None,
    max_topics=0,
    client=None,
    cache=None
):
    """Collect today's news, research it once, and create one daily PDF."""

    input_path = Path(input_file) if input_file is not None else DEFAULT_INPUT_FILE
    collection = run_pipeline(
        output_file=input_path,
        history_file=history_file
    )
    if not collection.saved_articles:
        raise ValueError(
            "No new articles were collected; no daily programme or PDF was created."
        )
    return run_research_report(
        input_file=input_path,
        brief_file=programme_file or DEFAULT_DAILY_OUTPUT_FILE,
        report_dir=report_dir,
        max_topics=max_topics,
        client=client,
        cache=cache
    )


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Collect AI news, research related coverage, and generate one "
            "combined daily programme and PDF."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_FILE)
    parser.add_argument(
        "--programme-output",
        type=Path,
        default=DEFAULT_DAILY_OUTPUT_FILE
    )
    parser.add_argument("--report-dir", type=Path, default=None)
    parser.add_argument("--history-file", type=Path, default=None)
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
        programme, programme_path, pdf_path = run_daily_programme(
            input_file=args.input,
            programme_file=args.programme_output,
            report_dir=args.report_dir,
            history_file=args.history_file,
            max_topics=args.limit
        )
    except GeminiClientError as error:
        parser.exit(
            2,
            f"AI research failed; no valid daily PDF was created: "
            f"{error}\n"
        )
    print(
        f"Daily pipeline complete for {date.today().isoformat()}: "
        f"researched stories={len(programme['stories'])}, "
        "combined programmes=2."
    )
    print("Daily programme JSON:", programme_path)
    print("Daily PDF:", pdf_path)
    return programme


if __name__ == "__main__":
    main()
