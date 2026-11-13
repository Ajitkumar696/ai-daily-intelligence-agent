
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple
from urllib.parse import urlsplit, urlunsplit
from zoneinfo import ZoneInfo


_STOPWORDS = frozenset(
    "about above after again also another because been before being between "
    "both could does during each from have having into just like more most "
    "other over such than that their them then there these they this those "
    "through under very were what when where which while will with would "
    "your today researchers introduced new".split()
)

_DISPLAY_CHAR_FIXES = {
    "\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2013": "-",
    "\u2014": " - ", "\u2212": "-", "\u00a0": " ", "\u202f": " ",
    "\u2009": " ", "\u200a": " ", "\u200b": "", "\ufeff": "",
}

_UNDRAWABLE = re.compile(
    "[\U0001F000-\U0001FFFF\u2600-\u27BF\u2B00-\u2BFF\uFE00-\uFE0F\u200d]"
)


def clean_display_text(text: Any) -> str:
    """Normalise text for on-screen display or speech."""
    value = str(text or "")
    for bad, good in _DISPLAY_CHAR_FIXES.items():
        value = value.replace(bad, good)
    value = _UNDRAWABLE.sub("", value)
    return re.sub(r"\s+", " ", value).strip()


def _shorten(text: Any, limit: int) -> str:
    value = clean_display_text(text)
    if len(value) <= limit:
        return value
    cut = value[:limit].rsplit(" ", 1)[0].rstrip(".,;:- ")
    return (cut or value[:limit]) + "..."


def _limit_words(text: Any, maximum: int) -> str:
    words = clean_display_text(text).split()
    if len(words) <= maximum:
        return " ".join(words)
    return " ".join(words[:maximum]).rstrip(".,;:-") + "."


def _tokens(text: Any) -> set:
    return {
        word for word in re.findall(r"[a-z0-9]{4,}", str(text or "").lower())
        if word not in _STOPWORDS
    }


def _story_tokens(story: Dict[str, Any]) -> set:
    parts = [
        story.get("headline", ""),
        story.get("topic", ""),
        story.get("summary", ""),
    ]
    for fact in story.get("key_facts") or []:
        if isinstance(fact, dict):
            parts.append(fact.get("statement", ""))
    return _tokens(" ".join(str(part) for part in parts))


def _match_story_index(
    narration: str,
    story_tokens: List[set],
    previous: int
) -> int:
    """Pick the story whose text the narration actually talks about."""
    scene_tokens = _tokens(narration)
    best, best_score = previous, 0

    for index, tokens in enumerate(story_tokens):
        score = len(scene_tokens & tokens)
        if score > best_score:
            best, best_score = index, score

    return best if best_score >= 2 else previous


def _format_srt_timestamp(seconds: float) -> str:
    """Format seconds into SRT timestamp format."""
    total_ms = int(round(seconds * 1000))
    hours = total_ms // 3600000
    minutes = (total_ms % 3600000) // 60000
    secs = (total_ms % 60000) // 1000
    ms = total_ms % 1000
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def apply_scene_durations(
    plan: Dict[str, Any],
    durations: List[float]
) -> Dict[str, Any]:
    """Rewrite scene timings, subtitles and totals."""
    if len(durations) != len(plan.get("scenes", [])):
        raise ValueError("One duration is required for every scene.")

    elapsed = 0.0
    srt_lines: List[str] = []
    subtitles: List[Dict[str, Any]] = []

    for index, (scene, duration) in enumerate(
        zip(plan["scenes"], durations), 1
    ):
        start = round(elapsed, 3)
        end = round(elapsed + float(duration), 3)
        elapsed = end

        scene["duration_seconds"] = round(float(duration), 3)
        scene["start_seconds"] = start
        scene["end_seconds"] = end

        srt_lines.append(
            f"{index}\n"
            f"{_format_srt_timestamp(start)} --> "
            f"{_format_srt_timestamp(end)}\n"
            f"{scene['narration']}\n"
        )

        subtitles.append({
            "index": index,
            "start_seconds": start,
            "end_seconds": end,
            "text": scene["narration"],
        })

    plan["subtitles_srt"] = "\n".join(srt_lines).strip()
    plan["subtitles"] = subtitles
    plan["actual_duration_seconds"] = round(elapsed, 3)
    return plan


def _split_into_sentences(text: str) -> List[str]:
    """Split text into sentences."""
    cleaned = text.strip()
    if not cleaned:
        return []

    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", cleaned)
        if sentence.strip()
    ]
    return sentences or [cleaned]


def _assign_durations(
    word_counts: List[int],
    target_duration: int
) -> List[int]:
    """Allocate scene durations while keeping the total exact."""
    if not word_counts:
        return []

    if target_duration < len(word_counts):
        raise ValueError(
            "Target duration is too short for the number of scenes."
        )

    total_words = sum(max(0, count) for count in word_counts)

    if total_words == 0:
        base = target_duration // len(word_counts)
        durations = [base] * len(word_counts)
        durations[-1] += target_duration - sum(durations)
        return durations

    durations = [
        max(1, int(round(
            (max(0, count) / total_words) * target_duration
        )))
        for count in word_counts
    ]

    # Correct rounding differences without allowing zero/negative scenes.
    while sum(durations) > target_duration:
        candidates = [
            i for i, value in enumerate(durations) if value > 1
        ]
        if not candidates:
            break
        index = max(candidates, key=lambda i: durations[i])
        durations[index] -= 1

    while sum(durations) < target_duration:
        index = max(
            range(len(durations)),
            key=lambda i: word_counts[i],
        )
        durations[index] += 1

    return durations


# ---------------------------------------------------------------------
# Three-minute source-aware planning helpers
# ---------------------------------------------------------------------

def _extract_text(value: Any, depth: int = 0) -> str:
    """Safely extract readable evidence from nested article data."""
    if value is None or depth > 4:
        return ""

    if isinstance(value, str):
        return value.strip()

    if isinstance(value, (int, float)):
        return str(value)

    if isinstance(value, list):
        parts = [_extract_text(item, depth + 1) for item in value]
        return " ".join(part for part in parts if part)

    if isinstance(value, dict):
        preferred_keys = (
            "text", "content", "summary", "description",
            "statement", "body", "extracted_content",
            "full_text", "article_text", "main_content",
        )
        parts = []

        for key in preferred_keys:
            if key in value:
                extracted = _extract_text(value[key], depth + 1)
                if extracted and extracted not in parts:
                    parts.append(extracted)

        return " ".join(parts)

    return ""


def _normalise_name(value: Any) -> str:
    return re.sub(
        r"[^a-z0-9]+",
        " ",
        str(value or "").lower()
    ).strip()


def _natural_join(values: List[str]) -> str:
    cleaned = [clean_display_text(value) for value in values if value]
    if len(cleaned) < 2:
        return cleaned[0] if cleaned else ""
    return ", ".join(cleaned[:-1]) + " and " + cleaned[-1]


def _article_source_names(article: Dict[str, Any]) -> set:
    names = set()

    for field in ("source", "publisher"):
        value = _normalise_name(article.get(field))
        if value:
            names.add(value)

    return names


def _report_sources(report: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Read source records from supported collection-report structures."""
    if not isinstance(report, dict):
        return []

    for key in ("sources", "source_results", "results"):
        records = report.get(key)
        if isinstance(records, list):
            return [
                item for item in records
                if isinstance(item, dict)
            ]

    return []


def _source_name(record: Dict[str, Any]) -> str:
    for key in ("source", "name", "publisher"):
        value = str(record.get(key) or "").strip()
        if value:
            return value
    return "Unknown source"


def _source_status(record: Dict[str, Any]) -> str:
    value = str(
        record.get("status")
        or record.get("collection_status")
        or ""
    ).strip().lower()

    if value in ("success", "successful", "ok", "completed"):
        return "success"

    if value in ("failed", "failure", "error"):
        return "failed"

    return value or "unknown"


def _reported_article_count(record: Dict[str, Any]) -> int:
    try:
        return int(record.get("article_count", 1) or 0)
    except (TypeError, ValueError):
        return 0


def _usable_article_text(article: Dict[str, Any]) -> str:
    """
    Return evidence-backed article text.

    A headline alone is not enough to create a narrated news summary.
    """
    summary = _extract_text(article.get("summary"))
    extracted = _extract_text(article.get("extracted_content"))
    content = summary or extracted

    if not content:
        return ""

    return clean_display_text(content)


def _article_title(article: Dict[str, Any]) -> str:
    return clean_display_text(
        article.get("title")
        or article.get("headline")
        or "Untitled article"
    )


def _article_url(article: Dict[str, Any]) -> str:
    value = article.get("url")

    if isinstance(value, str) and value.startswith(("https://", "http://")):
        return value

    return ""


def _article_source(article: Dict[str, Any]) -> str:
    return str(
        article.get("source")
        or article.get("publisher")
        or "Unknown source"
    ).strip()


def _is_excluded_source(name: Any) -> bool:
    """Exclude X/Twitter and Product Hunt as requested."""
    normalised = _normalise_name(name)

    return normalised in {
        "x",
        "twitter",
        "x twitter",
        "twitter x",
        "product hunt",
    }


def _article_recency(article: Dict[str, Any]) -> float:
    """
    Prefer discovery time over publication time.

    This allows an older article discovered in a new collection run
    to be considered, without pretending its publication date changed.
    """
    fields = (
        "discovered_at",
        "collected_at",
        "fetched_at",
        "ingested_at",
        "created_at",
        "published_at",
        "publication_date",
        "published",
        "date",
    )

    for field in fields:
        value = article.get(field)

        if value is None:
            continue

        try:
            if isinstance(value, (int, float)):
                return float(value)

            parsed = datetime.fromisoformat(
                str(value).strip().replace("Z", "+00:00")
            )

            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)

            return parsed.timestamp()

        except (TypeError, ValueError, OverflowError):
            continue

    return 0.0


def _article_score(article: Dict[str, Any]) -> float:
    """A small deterministic editorial score; evidence matters more than age."""
    evidence = _usable_article_text(article)
    title_and_text = f"{_article_title(article)} {evidence}".lower()
    relevance_terms = (
        "launch", "release", "research", "model", "benchmark", "open source",
        "developer", "business", "safety", "funding", "robot", "chip", "policy",
    )
    relevance = sum(term in title_and_text for term in relevance_terms)
    evidence_quality = min(len(evidence.split()) / 80.0, 1.0)
    full_text = 0.35 if article.get("content_source_type") == "full_article" else 0.0
    recency = _article_recency(article)
    age_days = max(0.0, (datetime.now(timezone.utc).timestamp() - recency) / 86400.0) if recency else 365.0
    freshness = max(0.0, 1.0 - age_days / 45.0)
    return relevance * 0.2 + evidence_quality * 2.0 + full_text + freshness * 0.5


def _match_report_status(article: Dict[str, Any], records: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Match the collector's configured source to source/publisher article fields."""
    article_names = [_normalise_name(article.get(key)) for key in ("source", "publisher")]
    article_names = [value for value in article_names if value]
    exact = [record for record in records if _normalise_name(_source_name(record)) in article_names]
    if len(exact) == 1:
        return exact[0]
    # A publisher sometimes omits a suffix (for example, "Anthropic"). Accept
    # only an unambiguous prefix match; never assign an article to two sources.
    partial = [
        record for record in records
        if any(
            min(len(name), len(_normalise_name(_source_name(record)))) >= 5
            and (name.startswith(_normalise_name(_source_name(record)))
                 or _normalise_name(_source_name(record)).startswith(name))
            for name in article_names
        )
    ]
    return partial[0] if len(partial) == 1 else None


def _canonical_article_url(article: Dict[str, Any]) -> str:
    url = _article_url(article)

    if not url:
        return ""

    try:
        parts = urlsplit(url)
        return urlunsplit((
            parts.scheme.lower(),
            parts.netloc.lower().removeprefix("www."),
            parts.path.rstrip("/"),
            "",
            "",
        ))
    except ValueError:
        return url.rstrip("/").lower()


def _make_article_narration(
    article: Dict[str, Any],
    max_words: int = 24,
    transition: str = "",
) -> str:
    """Create concise narration from collected article evidence only."""
    source = clean_display_text(_article_source(article))
    title = _article_title(article)
    evidence = _usable_article_text(article)

    if not evidence:
        return ""

    sentences = _split_into_sentences(evidence)
    evidence_text = " ".join(sentences[:2]) if sentences else evidence

    lead = f"{transition} {source}:" if transition else f"From {source}:"
    narration = f"{lead} {title}. {evidence_text}"
    return _limit_words(narration, max_words)


def _source_article_groups(
    articles: List[Dict[str, Any]],
    report: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], List[str], List[str], List[str]]:
    """
    Prioritise one recent usable article from every publisher first.
    Add further recent stories only when the narration budget permits.
    """
    report_records = _report_sources(report)
    has_report = bool(report_records)
    success_records = [
        record for record in report_records
        if _source_status(record) == "success"
        and _reported_article_count(record) > 0
    ]
    failed_sources = [
        _source_name(record) for record in report_records
        if _source_status(record) == "failed"
    ]
    no_article_sources = [
        _source_name(record) for record in report_records
        if _source_status(record) == "empty"
        or (_source_status(record) == "success"
            and _reported_article_count(record) <= 0)
    ]
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    seen_urls = set()
    seen_titles = set()
    seen_title_tokens: List[set] = []

    for article in articles:
        if not isinstance(article, dict):
            continue

        source = _article_source(article)
        source_key = _normalise_name(source)

        report_match = _match_report_status(article, success_records) if has_report else None
        if has_report and report_match is None:
            continue
        if report_match is not None:
            source = _source_name(report_match)
            source_key = _normalise_name(source)

        if not source_key or _is_excluded_source(source):
            continue

        if any(
            _is_excluded_source(name)
            for name in _article_source_names(article)
        ):
            continue

        if not _usable_article_text(article):
            continue

        canonical_url = _canonical_article_url(article)
        title_key = _normalise_name(_article_title(article))
        title_tokens = _tokens(title_key)

        if canonical_url and canonical_url in seen_urls:
            continue

        title_duplicate_key = (source_key, title_key)
        similar_title = any(
            len(title_tokens) >= 6 and len(prior) >= 6
            and len(title_tokens & prior) >= 5
            and len(title_tokens & prior) / len(title_tokens | prior) >= 0.82
            for prior in seen_title_tokens
        )
        if title_duplicate_key in seen_titles or similar_title:
            continue

        if canonical_url:
            seen_urls.add(canonical_url)

        seen_titles.add(title_duplicate_key)
        if title_tokens:
            seen_title_tokens.append(title_tokens)
        grouped.setdefault(source_key, []).append(article)

    for records in grouped.values():
        records.sort(key=_article_score, reverse=True)

    if not grouped:
        return [], failed_sources, no_article_sources, [
            _source_name(record) for record in success_records
        ]

    # Keep the newest first article from each usable publisher.
    first_articles = [
        records[0]
        for records in grouped.values()
        if records
    ]
    first_articles.sort(key=_article_score, reverse=True)

    # Reserve room for intro/outro and narration pacing.
    story_budget = 335
    source_count = len(first_articles)
    words_per_source = max(
        12,
        min(42, story_budget // max(source_count, 1)),
    )

    selected = []
    current_words = 0

    # First pass: publisher diversity.
    for article in first_articles:
        narration = _make_article_narration(
            article,
            max_words=words_per_source,
        )
        if not narration:
            continue

        selected.append(article)
        current_words += len(narration.split())

    # Second pass: add further stories, newest first.
    extras = []

    for records in grouped.values():
        extras.extend(records[1:])

    extras.sort(key=_article_score, reverse=True)

    for article in extras:
        narration = _make_article_narration(article, max_words=42)
        word_count = len(narration.split())

        if not narration or current_words + word_count > story_budget:
            continue

        selected.append(article)
        current_words += word_count

    selected_sources = {
        _normalise_name(_source_name(match))
        for item in selected
        if (match := _match_report_status(item, success_records)) is not None
    }
    uncovered = [
        _source_name(record) for record in success_records
        if _normalise_name(_source_name(record)) not in selected_sources
    ]
    return selected, failed_sources, no_article_sources, uncovered


def _make_source_scene(
    index: int,
    article: Dict[str, Any],
    narration: str
) -> Dict[str, Any]:
    source = _article_source(article)
    title = _article_title(article)
    url = _article_url(article)
    summary = _usable_article_text(article)

    return {
        "scene_number": index,
        "kind": "story",
        "title": title,
        "headline": title,
        "topic": source,
        "key_points": [_shorten(summary, 150)],
        "label": f"Source: {source}",
        "narration": narration,
        "duration_seconds": 0,
        "start_seconds": 0,
        "end_seconds": 0,
        "visual_suggestion": (
            "News visual illustrating the verified article: "
            f"{_shorten(title, 100)}"
        ),
        "on_screen_text": f"{title}\n{source}",
        "source_urls": [url] if url else [],
        "source_credit": source,
        "story_id": str(article.get("story_id") or ""),
    }


def _make_status_scene(
    index: int,
    narration: str,
    title: str,
    label: str,
    kind: str = "outro",
) -> Dict[str, Any]:
    return {
        "scene_number": index,
        "kind": kind,
        "title": title,
        "headline": title,
        "topic": "Collection status",
        "key_points": [],
        "label": label,
        "narration": narration,
        "duration_seconds": 0,
        "start_seconds": 0,
        "end_seconds": 0,
        "visual_suggestion": "Clean AI news briefing graphic.",
        "on_screen_text": title,
        "source_urls": [],
        "source_credit": "",
        "story_id": "",
    }


def _plan_three_minute_from_sources(
    programme: Dict[str, Any],
    selected: Dict[str, Any],
    programme_date: str,
    title: str,
    target_duration: int,
) -> Dict[str, Any]:
    """Build a three-minute plan from recent, usable collected articles."""
    articles = programme.get("source_articles") or []

    if not isinstance(articles, list):
        articles = []

    articles = [
        article for article in articles
        if isinstance(article, dict)
    ]

    selected_articles, failed_sources, no_article_sources, uncovered_sources = _source_article_groups(
        articles,
        programme.get("collection_report") or {},
    )

    # Refresh this timestamp on every planning run.
    generated_at = datetime.now(
        ZoneInfo("Asia/Kolkata")
    ).strftime("%d %b %Y, %I:%M %p IST")

    intro = (
        "Welcome to AI Daily Intelligence. "
        "Here are the latest verified developments across the AI ecosystem."
    )

    scenes = [
        _make_status_scene(
            1,
            intro,
            title,
            "Introduction",
            kind="intro",
        )
    ]

    source_count = len({
        _normalise_name(_article_source(article))
        for article in selected_articles
    })

    words_per_source = max(18, min(42, 335 // max(source_count, 1)))

    transitions = (
        "First,", "Next,", "Elsewhere,", "Separately,",
        "Another development:", "In a related update,", "Also,",
        "Turning to another story,", "From another corner of AI,",
        "A further update:", "Finally,",
    )
    for article_index, article in enumerate(selected_articles):
        narration = _make_article_narration(
            article,
            max_words=words_per_source,
            transition=transitions[article_index % len(transitions)],
        )

        if narration:
            scenes.append(
                _make_source_scene(
                    len(scenes) + 1,
                    article,
                    narration,
                )
            )

    if not selected_articles:
        scenes.append(
            _make_status_scene(
                len(scenes) + 1,
                "No usable source articles were available for this briefing. "
                "The video does not invent or infer missing news.",
                "No usable articles available",
                "Collection information",
            )
        )

    if failed_sources:
        unavailable = _natural_join(failed_sources)
        scenes.append(_make_status_scene(
            len(scenes) + 1,
            f"Collection notice: {unavailable} were unavailable during collection.",
            "Sources unavailable",
            "Collection status",
        ))
    no_coverage = list(dict.fromkeys(no_article_sources + uncovered_sources))
    if no_coverage:
        names = _natural_join(no_coverage)
        scenes.append(_make_status_scene(
            len(scenes) + 1,
            f"No usable article content was available from {names} for this briefing.",
            "Sources without usable coverage",
            "Collection status",
        ))

    outro = (
        "That concludes this AI Daily Intelligence briefing. "
        "Check the daily report for original articles and source links."
    )

    scenes.append(
        _make_status_scene(
            len(scenes) + 1,
            outro,
            "That's your AI briefing",
            "Closing and sign-off",
        )
    )

    # Guarantee a safe minimum duration for every scene.
    if target_duration < len(scenes):
        raise ValueError(
            f"Target duration ({target_duration}s) is too short "
            f"for {len(scenes)} scenes."
        )

    narration = " ".join(
        scene["narration"] for scene in scenes
    ).strip()

    durations = _assign_durations(
        [len(scene["narration"].split()) for scene in scenes],
        target_duration,
    )

    report_records = _report_sources(programme.get("collection_report") or {})
    covered_sources = list(dict.fromkeys(
        _source_name(match) if match else _article_source(article)
        for article in selected_articles
        for match in [_match_report_status(article, report_records)]
    ))

    plan = {
        "programme_date": programme_date,
        "generated_at": generated_at,
        "programme_type": "three_minute",
        "title": title,
        "target_duration_seconds": target_duration,
        "actual_duration_seconds": 0,
        "narration": narration,
        "word_count": len(narration.split()),
        "scene_count": len(scenes),
        "scenes": scenes,
        "subtitles_srt": "",
        "subtitles": [],
        "stories": [],
        "source_coverage": {
            "usable_source_count": len(covered_sources),
            "selected_article_count": len(selected_articles),
            "covered_sources": covered_sources,
            "failed_sources": failed_sources,
            "sources_without_articles": no_article_sources,
            "sources_without_usable_coverage": uncovered_sources,
        },
    }

    plan = apply_scene_durations(plan, durations)

    for index, scene in enumerate(plan["scenes"], 1):
        scene["scene_number"] = index

    plan["scene_count"] = len(plan["scenes"])
    return plan


# ---------------------------------------------------------------------
# Original programme-based planner
# ---------------------------------------------------------------------

def plan_video_scenes(
    programme: Dict[str, Any],
    programme_type: str = "one_minute"
) -> Dict[str, Any]:
    """
    Plan synchronized video scenes.

    One-minute videos continue using the original daily programme
    narration. Source-aware planning is exclusive to three-minute videos.
    """
    if not isinstance(programme, dict):
        raise ValueError("Programme must be a dictionary.")

    programme_date = programme.get("programme_date", "")
    programmes = programme.get("programmes", {})
    stories = programme.get("stories", [])

    if not isinstance(programmes, dict):
        raise ValueError("Programme 'programmes' must be a dictionary.")

    if programme_type not in programmes:
        raise ValueError(
            f"Programme type '{programme_type}' not found in daily programme. "
            f"Available types: {list(programmes.keys())}"
        )

    selected = programmes[programme_type]

    if not isinstance(selected, dict):
        raise ValueError(
            f"Programme entry '{programme_type}' must be a dictionary."
        )

    target_duration = int(
        selected.get(
            "duration_seconds",
            60 if programme_type == "one_minute" else 180,
        )
    )

    title = str(
        selected.get(
            "title",
            f"Daily AI Intelligence - {programme_date}",
        )
    )

    # Supplemental collection data may be attached by the video entry point.
    # When a validated programme already has stories, keep the three-minute
    # video on that same editorial package as the PDF. Source-only planning is
    # reserved for legacy programmes that have no researched story records.
    if (
        programme_type == "three_minute"
        and isinstance(programme.get("source_articles"), list)
        and not any(isinstance(story, dict) for story in stories)
    ):
        return _plan_three_minute_from_sources(
            programme,
            selected,
            programme_date,
            title,
            target_duration,
        )

    # Original programme-based planner: retained for one-minute videos.
    selected_story_ids = selected.get("story_ids")

    if isinstance(selected_story_ids, list):
        selected_story_ids = set(selected_story_ids)
        stories = [
            story for story in stories
            if isinstance(story, dict)
            and story.get("story_id") in selected_story_ids
        ]

    narration = str(selected.get("narration", "")).strip()

    if not narration:
        raise ValueError(
            f"No narration found for '{programme_type}'."
        )

    sentences = _split_into_sentences(narration)
    n = len(sentences)

    story_token_sets = [
        _story_tokens(story) for story in stories
    ]

    max_sentences_per_scene = 2
    scene_sentence_groups: List[Tuple[str, List[str], Any]] = []

    if n <= 3:
        for idx, sentence in enumerate(sentences, 1):
            scene_sentence_groups.append(
                (f"Scene {idx}", [sentence], None)
            )
    else:
        matches = [
            _match_story_index(sentence, story_token_sets, None)
            if stories else None
            for sentence in sentences
        ]

        lead = 0
        while lead < n and matches[lead] is None:
            lead += 1

        tail = 0
        while tail < n - lead and matches[-1 - tail] is None:
            tail += 1

        if not stories or lead + tail >= n:
            lead = tail = (1 if n <= 6 else 2)
            matches = [None] * n

        lead, tail = min(lead, 3), min(tail, 3)

        mid_matches = matches[lead:n - tail]
        mid = sentences[lead:n - tail]

        filled: List[int] = []
        previous_story = next(
            (match for match in mid_matches if match is not None),
            0,
        )

        for match in mid_matches:
            previous_story = (
                previous_story if match is None else match
            )
            filled.append(previous_story)

        if lead:
            scene_sentence_groups.append(
                ("Introduction", sentences[:lead], None)
            )

        runs: List[List[Any]] = []

        for sentence, story_index in zip(mid, filled):
            if runs and runs[-1][0] == story_index:
                runs[-1][1].append(sentence)
            else:
                runs.append([story_index, [sentence]])

        for story_index, run in runs:
            for i in range(0, len(run), max_sentences_per_scene):
                scene_sentence_groups.append((
                    f"Story {len(scene_sentence_groups) + 1}",
                    run[i:i + max_sentences_per_scene],
                    story_index,
                ))

        if tail:
            scene_sentence_groups.append(
                ("Closing & Sign-Off", sentences[n - tail:], None)
            )

    primary_urls = selected.get("source_urls", [])

    if not primary_urls and stories:
        primary_urls = stories[0].get("source_urls", [])

    scene_word_counts = [
        sum(len(sentence.split()) for sentence in sentence_list)
        for _, sentence_list, _ in scene_sentence_groups
    ]

    durations = _assign_durations(
        scene_word_counts,
        target_duration,
    )

    last_story_idx = 0
    scenes: List[Dict[str, Any]] = []
    subtitles: List[Dict[str, Any]] = []
    srt_lines: List[str] = []
    current_time = 0.0

    for idx, (
        (label, sentence_list, assigned_story_idx),
        duration,
    ) in enumerate(zip(scene_sentence_groups, durations), 1):

        scene_narration = " ".join(sentence_list)
        start_time = current_time
        end_time = round(start_time + duration, 2)
        current_time = end_time

        if stories:
            story_idx = assigned_story_idx

            if story_idx is None:
                story_idx = _match_story_index(
                    scene_narration,
                    story_token_sets,
                    last_story_idx,
                )

            last_story_idx = story_idx
            matched_story = stories[story_idx]
        else:
            matched_story = {}

        story_urls = matched_story.get(
            "source_urls",
            primary_urls,
        )

        source_records = matched_story.get("sources", [])
        credits = []

        for source in source_records:
            if not isinstance(source, dict):
                continue

            url = source.get("url")
            domain = (
                url.split("/", 3)[2].removeprefix("www.")
                if isinstance(url, str) and "://" in url
                else ""
            )

            publisher = str(
                source.get("publisher") or ""
            ).strip()

            credit = (
                f"{publisher} ({domain})"
                if publisher and domain
                else publisher or domain
            )

            if credit and credit not in credits:
                credits.append(credit)

        if not credits:
            credits = list(dict.fromkeys(
                url.split("/", 3)[2].removeprefix("www.")
                for url in story_urls
                if isinstance(url, str) and "://" in url
            ))

        source_credit = " · ".join(credits)

        story_headline = clean_display_text(
            matched_story.get("headline", title)
        )

        story_topic = clean_display_text(
            matched_story.get("topic", "")
        )

        small = n <= 3
        is_intro = (
            label == "Introduction"
            or (small and idx == 1)
        )
        is_outro = (
            label == "Closing & Sign-Off"
            or (small and idx == len(scene_sentence_groups))
        )

        if is_intro:
            kind = "intro"
            scene_title = title
            visual = (
                "Anchor graphic with futuristic AI news backdrop "
                "and daily date badge."
            )
            key_points = []
            on_screen = (
                f"AI DAILY INTELLIGENCE\n{programme_date}\n{title}"
            )

        elif is_outro:
            kind = "outro"
            scene_title = "That's your AI briefing for today"
            visual = (
                "Summary graphic with verified source checklist."
            )
            key_points = [
                "Full report and papers are in today's PDF"
            ]
            on_screen = (
                "FULL REPORT & PAPERS AVAILABLE IN DAILY PDF"
            )

        else:
            kind = "story"
            scene_title = story_topic or story_headline
            visual = (
                "Diagram illustrating: "
                f"{_shorten(matched_story.get('why_it_matters', scene_narration), 70)}"
            )

            key_points = [
                _shorten(fact.get("statement", ""), 150)
                for fact in (
                    matched_story.get("key_facts") or []
                )[:3]
                if isinstance(fact, dict)
                and fact.get("statement")
            ]

            if (
                not key_points
                and matched_story.get("why_it_matters")
            ):
                key_points = [
                    _shorten(
                        matched_story["why_it_matters"],
                        150,
                    )
                ]

            on_screen = "\n".join(
                [story_headline] + key_points
            )

        if kind != "story":
            story_urls = []
            source_credit = ""

        scene_obj = {
            "scene_number": idx,
            "kind": kind,
            "title": scene_title,
            "headline": (
                story_headline if kind == "story"
                else scene_title
            ),
            "topic": story_topic,
            "key_points": key_points,
            "label": label,
            "narration": scene_narration,
            "duration_seconds": duration,
            "start_seconds": start_time,
            "end_seconds": end_time,
            "visual_suggestion": visual,
            "on_screen_text": on_screen,
            "source_urls": story_urls,
            "source_credit": source_credit,
            "story_id": (
                str(matched_story.get("story_id", ""))
                if kind == "story" else ""
            ),
        }

        scenes.append(scene_obj)

        srt_lines.append(
            f"{idx}\n"
            f"{_format_srt_timestamp(start_time)} --> "
            f"{_format_srt_timestamp(end_time)}\n"
            f"{scene_narration}\n"
        )

        subtitles.append({
            "index": idx,
            "start_seconds": start_time,
            "end_seconds": end_time,
            "text": scene_narration,
        })

    reconstructed = " ".join(
        scene["narration"] for scene in scenes
    ).strip()

    if " ".join(reconstructed.split()) != " ".join(narration.split()):
        raise AssertionError(
            "Scene planning altered the original narration text!"
        )

    return {
        "programme_date": programme_date,
        "programme_type": programme_type,
        "title": title,
        "target_duration_seconds": target_duration,
        "actual_duration_seconds": sum(
            scene["duration_seconds"] for scene in scenes
        ),
        "narration": narration,
        "word_count": len(narration.split()),
        "scene_count": len(scenes),
        "scenes": scenes,
        "subtitles_srt": "\n".join(srt_lines).strip(),
        "subtitles": subtitles,
        "stories": stories,
    }
