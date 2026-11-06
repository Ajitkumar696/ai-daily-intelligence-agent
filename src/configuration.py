from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union
from urllib.parse import urlsplit

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCES_FILE = PROJECT_ROOT / "config" / "sources.yaml"
SUPPORTED_SOURCE_TYPES = {
    "rss",
    "api",
    "sitemap",
    "api_hf_spaces",
    "twitter",
    "producthunt",
}
REQUIRED_SOURCE_FIELDS = {"name", "type", "url", "max_articles"}

VALID_CONTENT_TYPES = {
    "article",
    "newsletter",
    "podcast",
    "paper",
    "space",
    "video",
    "social",
    "launch",
}


@dataclass(frozen=True)
class SourceConfig:
    name: str
    type: str
    url: str
    max_articles: int
    extract_articles: bool = False
    crawl4ai_fallback: bool = False
    crawl4ai_allowed_domains: tuple = ()
    # Classifies what kind of content this source produces.
    # Allowed values: article | newsletter | podcast | paper | space | video
    content_type: str = "article"


def load_sources_config(
    config_path: Optional[Union[str, Path]] = None
) -> List[SourceConfig]:
    """Load and validate configured article sources."""

    path = Path(config_path) if config_path is not None else DEFAULT_SOURCES_FILE

    if not path.is_file():
        raise FileNotFoundError(
            f"Sources configuration file not found: {path}"
        )

    try:
        with path.open("r", encoding="utf-8") as file:
            config = yaml.safe_load(file)
    except yaml.YAMLError as error:
        raise ValueError(
            f"Malformed YAML in sources configuration '{path}': {error}"
        ) from error

    if not isinstance(config, dict):
        raise ValueError(
            f"Sources configuration '{path}' must contain a YAML mapping."
        )

    sources = config.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError(
            f"Sources configuration '{path}' must contain a non-empty "
            "'sources' list."
        )

    validated_sources = []

    for index, source in enumerate(sources, start=1):
        if not isinstance(source, dict):
            raise ValueError(
                f"Source {index} in '{path}' must be a YAML mapping."
            )

        missing_fields = REQUIRED_SOURCE_FIELDS - source.keys()
        if missing_fields:
            missing = ", ".join(sorted(missing_fields))
            raise ValueError(
                f"Source {index} in '{path}' is missing required fields: "
                f"{missing}."
            )

        name = source["name"]
        source_type = source["type"]
        url = source["url"]
        max_articles = source["max_articles"]
        extract_articles = source.get("extract_articles", False)
        crawl4ai_fallback = source.get("crawl4ai_fallback", False)
        crawl4ai_allowed_domains = source.get("crawl4ai_allowed_domains", [])
        content_type = source.get("content_type", "article")

        if (
            not isinstance(source_type, str)
            or source_type not in SUPPORTED_SOURCE_TYPES
        ):
            supported = ", ".join(sorted(SUPPORTED_SOURCE_TYPES))
            raise ValueError(
                f"Unsupported source type {source_type!r} for source "
                f"{name!r}. Supported types: {supported}."
            )

        if not isinstance(name, str) or not name.strip():
            raise ValueError(
                f"Source {index} in '{path}' must have a non-empty name."
            )

        if not isinstance(url, str) or not url.strip():
            raise ValueError(
                f"Source {name!r} in '{path}' must have a non-empty URL."
            )

        if (
            not isinstance(max_articles, int)
            or isinstance(max_articles, bool)
            or max_articles <= 0
        ):
            raise ValueError(
                f"Source {name!r} in '{path}' must have a positive integer "
                "'max_articles' value."
            )

        if not isinstance(extract_articles, bool):
            raise ValueError(
                f"Source {name!r} in '{path}' must have a boolean "
                "'extract_articles' value."
            )

        if extract_articles and source_type not in ("rss", "sitemap"):
            raise ValueError(
                f"Article extraction can only be enabled for RSS or sitemap "
                f"sources: {name!r}."
            )

        if not isinstance(crawl4ai_fallback, bool):
            raise ValueError(
                f"Source {name!r} in '{path}' must have a boolean "
                "'crawl4ai_fallback' value."
            )
        if not isinstance(crawl4ai_allowed_domains, list) or any(
            not isinstance(domain, str)
            or not domain.strip()
            or urlsplit("//" + domain.strip()).hostname != domain.strip().lower()
            for domain in crawl4ai_allowed_domains
        ):
            raise ValueError(
                f"Source {name!r} in '{path}' must provide valid hostnames "
                "in 'crawl4ai_allowed_domains'."
            )
        if crawl4ai_fallback and (
            not extract_articles or source_type != "rss"
            or not crawl4ai_allowed_domains
        ):
            raise ValueError(
                f"Crawl4AI fallback for {name!r} requires RSS article "
                "extraction and an explicit allowed-domain list."
            )

        if content_type not in VALID_CONTENT_TYPES:
            raise ValueError(
                f"Source {name!r} in '{path}' has an unsupported "
                f"'content_type': {content_type!r}. "
                f"Allowed values: {', '.join(sorted(VALID_CONTENT_TYPES))}."
            )

        validated_sources.append(
            SourceConfig(
                name=name.strip(),
                type=source_type,
                url=url.strip(),
                max_articles=max_articles,
                extract_articles=extract_articles,
                crawl4ai_fallback=crawl4ai_fallback,
                crawl4ai_allowed_domains=tuple(
                    domain.strip().lower() for domain in crawl4ai_allowed_domains
                ),
                content_type=content_type,
            )
        )

    return validated_sources
