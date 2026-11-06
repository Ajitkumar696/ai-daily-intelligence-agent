
from datetime import datetime, timezone
from bs4 import BeautifulSoup


def normalize_article(source, entry):
    """
    Convert an RSS entry into our common article structure.
    """

    # Extract basic information
    title = entry.get("title", "").strip()
    url = entry.get("link", "").strip()
    published = (
        entry["published"]
        if "published" in entry
        else (entry["updated"] if "updated" in entry else "")
    )
    summary = entry.get("summary", "")

    # Skip entries without a title or URL
    if not title or not url:
        return None

    # Remove HTML tags from the summary
    clean_summary = BeautifulSoup(
        summary, "html.parser"
    ).get_text(" ", strip=True)
    author = entry.get("author", "")
    tags = entry.get("tags", [])
    normalized_tags = [
        {
            "term": tag.get("term", ""),
            "label": tag.get("label", "")
        }
        for tag in tags
        if isinstance(tag, dict)
    ] if isinstance(tags, list) else []

    # Create standardized article dictionary
    article = {
        "source": source,
        "publisher": source,
        "source_type": "rss",
        "title": title,
        "url": url,
        "published": published,
        "author": author.strip() if isinstance(author, str) else "",
        "summary": clean_summary,
        "metadata": {
            "id": entry.get("id", ""),
            "updated": entry["updated"] if "updated" in entry else "",
            "tags": normalized_tags
        },
        "collected_at": datetime.now(timezone.utc).isoformat()
    }

    return article
