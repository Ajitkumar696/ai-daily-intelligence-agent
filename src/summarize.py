import argparse
from pathlib import Path

from src.summarization.gemini_client import GeminiClientError
from src.summarization.providers import check_groq_configuration
from src.summarization.summarizer import (
    DEFAULT_INPUT_FILE,
    DEFAULT_OUTPUT_FILE,
    get_max_articles_per_run,
    load_articles,
    save_summaries,
    summarize_articles
)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Summarize collected articles using the configured Groq model."
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
        default=DEFAULT_OUTPUT_FILE,
        help="Structured summary JSON output."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum articles; 0 processes all articles."
    )
    parser.add_argument(
        "--check-config",
        action="store_true",
        help="Check the selected LLM configuration without making an API request."
    )
    args = parser.parse_args(argv)
    if args.check_config:
        try:
            status = check_groq_configuration()
        except GeminiClientError as error:
            print(f"Groq configuration invalid: {error}")
            return 1
        print(
            f"{status['provider'].title()} configuration: "
            "API key configured=yes, "
            f"model={status['model']}, "
            f"base_url={status['base_url']}."
        )
        for issue in status["issues"]:
            print(f"Configuration issue: {issue}")
        if status["ready"]:
            print("Configuration checks passed. No API request was made.")
            return 0
        print("Configuration checks failed. No API request was made.")
        return 1

    limit = get_max_articles_per_run(args.limit)
    if limit < 0:
        parser.error("--limit must be zero or a positive integer.")

    articles = load_articles(args.input)
    try:
        summaries = summarize_articles(articles, max_articles=limit)
    except GeminiClientError as error:
        parser.exit(
            2,
            f"AI provider setup failed before any API requests "
            f"(API calls=0): {error}\n"
        )
    output = save_summaries(summaries, args.output)

    successes = sum(
        item["summary_status"] == "SUCCESS"
        for item in summaries
    )
    insufficient = sum(
        item["summary_status"] == "INSUFFICIENT_CONTENT"
        for item in summaries
    )
    failures = sum(item["summary_status"] == "FAILED" for item in summaries)
    print(
        f"Summarization complete: {successes} successful, "
        f"{insufficient} insufficient, {failures} failed. Saved: {output}"
    )
    return summaries


if __name__ == "__main__":
    result = main()
    if isinstance(result, int):
        raise SystemExit(result)
