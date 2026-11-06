import argparse
from collections import Counter
from pathlib import Path

from src.reporting.pdf_generator import generate_pdf_report
from src.summarization.summarizer import (
    DEFAULT_INPUT_FILE,
    DEFAULT_OUTPUT_FILE,
    get_max_articles_per_run,
    load_articles,
    save_summaries,
    summarize_articles
)


def run_daily_report(
    input_file=None,
    summarized_file=None,
    report_dir=None,
    max_articles=None
):
    """Summarize collected articles, persist JSON, then generate a PDF."""

    articles = load_articles(input_file or DEFAULT_INPUT_FILE)
    summaries = summarize_articles(
        articles,
        max_articles=get_max_articles_per_run(max_articles)
    )
    summarized_path = save_summaries(
        summaries,
        summarized_file or DEFAULT_OUTPUT_FILE
    )

    counts = Counter(item["summary_status"] for item in summaries)
    try:
        pdf_path = generate_pdf_report(summaries, output_dir=report_dir)
    except Exception as error:
        print(
            "Daily report execution incomplete: PDF generation failed "
            f"({type(error).__name__}). Summarized JSON was preserved at "
            f"{summarized_path}."
        )
        raise

    print(
        "Daily report execution complete: "
        f"processed={len(summaries)}, "
        f"successful={counts.get('SUCCESS', 0)}, "
        f"insufficient={counts.get('INSUFFICIENT_CONTENT', 0)}, "
        f"failed={counts.get('FAILED', 0)}"
    )
    print("Summarized JSON:", summarized_path)
    print("PDF report:", pdf_path)
    return summaries, summarized_path, pdf_path


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Summarize collected articles and generate a PDF report."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_FILE
    )
    parser.add_argument(
        "--summarized-output",
        type=Path,
        default=DEFAULT_OUTPUT_FILE
    )
    parser.add_argument("--report-dir", type=Path, default=None)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum articles; 0 processes all."
    )
    args = parser.parse_args(argv)
    run_daily_report(
        input_file=args.input,
        summarized_file=args.summarized_output,
        report_dir=args.report_dir,
        max_articles=args.limit
    )


if __name__ == "__main__":
    main()
