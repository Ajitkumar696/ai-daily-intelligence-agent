
import json
from pathlib import Path

from .deduplicate import canonicalize_url
from .json_storage import atomic_write_json


PROJECT_ROOT = Path(__file__).resolve().parents[2]

HISTORY_FILE = PROJECT_ROOT / "output" / "processed_history.json"


def load_history(history_file=None):
    """Load URLs of previously reported articles."""

    path = Path(history_file) if history_file is not None else HISTORY_FILE

    if not path.exists():
        return set()

    try:
        with path.open("r", encoding="utf-8") as file:
            urls = json.load(file)
    except (OSError, ValueError) as error:
        raise ValueError(f"Unable to read valid history from '{path}': {error}") from error

    if (
        not isinstance(urls, list)
        or any(not isinstance(url, str) or not url.strip() for url in urls)
    ):
        raise ValueError(
            f"History file '{path}' must contain a JSON list of non-empty "
            "URL strings."
        )

    return {canonicalize_url(url) for url in urls}


def filter_previously_reported(articles, history_file=None):
    """Remove articles that have already been reported."""

    reported_urls = load_history(history_file)

    new_articles = [
        article
        for article in articles
        if canonicalize_url(article.get("url", "")) not in reported_urls
    ]

    skipped = len(articles) - len(new_articles)

    print("\nHISTORY CHECK")
    print("Previously reported:", skipped)
    print("New articles:", len(new_articles))

    return new_articles


def mark_as_reported(articles, history_file=None):
    """Save article URLs after they have been included in a report."""

    path = Path(history_file) if history_file is not None else HISTORY_FILE
    reported_urls = load_history(path)

    for article in articles:
        url = article.get("url", "").strip()

        if url:
            reported_urls.add(canonicalize_url(url))

    atomic_write_json(path, sorted(reported_urls))

    print("[OK] Report history updated:", path)
