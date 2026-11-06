import argparse
import json
from pathlib import Path

from src.reporting.pdf_generator import generate_pdf_report
from src.summarization.summarizer import DEFAULT_OUTPUT_FILE


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Generate a PDF report from summaries or successful research "
            "briefs."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_OUTPUT_FILE,
        help="Summary JSON or research brief JSON file."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="PDF output directory (defaults to output/reports)."
    )
    args = parser.parse_args(argv)

    with args.input.open("r", encoding="utf-8") as file:
        summaries = json.load(file)
    if not isinstance(summaries, list):
        parser.error("Summary input must contain a JSON list.")

    output_path = generate_pdf_report(summaries, output_dir=args.output_dir)
    print("PDF report generated:", output_path)
    return output_path


if __name__ == "__main__":
    main()
