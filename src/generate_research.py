import argparse
from pathlib import Path

from src.processing.json_storage import atomic_write_json
from src.research.daily_programme import generate_daily_programme
from src.research.grouping import group_related_articles
from src.summarization.gemini_client import GeminiClientError
from src.summarization.summarizer import (
    DEFAULT_INPUT_FILE,
    load_articles
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DAILY_OUTPUT_FILE = PROJECT_ROOT / "output" / "daily_programme.json"


def run_research(
    input_file=None,
    output_file=None,
    max_topics=0,
    client=None,
    cache=None,
    programme_date=None
):
    articles = load_articles(input_file or DEFAULT_INPUT_FILE)
    groups = group_related_articles(articles)
    if max_topics:
        groups = groups[:max_topics]
    programme = generate_daily_programme(
        groups,
        client=client,
        cache=cache,
        programme_date=programme_date
    )
    output = (
        Path(output_file)
        if output_file is not None
        else DEFAULT_DAILY_OUTPUT_FILE
    )
    atomic_write_json(output, programme)
    return groups, programme, output


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Research grouped coverage and create one combined daily news "
            "programme with one-minute and three-minute scripts."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_FILE,
        help="Collected article JSON file."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_DAILY_OUTPUT_FILE,
        help="Combined daily programme JSON output."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help=(
            "Maximum grouped topics to send to the AI provider; 0 includes all "
            "available groups."
        )
    )
    args = parser.parse_args(argv)
    if args.limit < 0:
        parser.error("--limit must be zero or a positive integer.")
    try:
        groups, programme, output = run_research(
            args.input,
            args.output,
            max_topics=args.limit
        )
    except GeminiClientError as error:
        parser.exit(
            2,
            f"AI research failed; no valid daily programme was saved: "
            f"{error}\n"
        )

    print(
        f"Daily programme complete: available groups={len(groups)}, "
        f"selected stories={len(programme['stories'])}. Saved: {output}"
    )
    return programme


if __name__ == "__main__":
    result = main()
    if isinstance(result, int):
        raise SystemExit(result)
