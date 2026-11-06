"""Per-story visual facts: keywords, real statistics, source names, images.

Nothing here invents content. Statistics are read from the story's own text;
images come only from files the user supplied (or an explicit local path in
the story data) - nothing is scraped from the web, which keeps the pipeline
clear of copyright, paywall and licence problems.
"""
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from urllib.parse import urlsplit

from .planner import clean_display_text

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp")

_KNOWN_SOURCES = {
    "huggingface.co": "Hugging Face", "arxiv.org": "arXiv", "openai.com": "OpenAI",
    "anthropic.com": "Anthropic", "deepmind.google": "Google DeepMind",
    "blog.google": "Google", "ai.googleblog.com": "Google AI Blog",
    "research.google": "Google Research", "ai.meta.com": "Meta AI",
    "techcrunch.com": "TechCrunch", "theverge.com": "The Verge",
    "venturebeat.com": "VentureBeat", "mit.edu": "MIT", "nvidia.com": "NVIDIA",
    "microsoft.com": "Microsoft", "github.com": "GitHub", "producthunt.com": "Product Hunt",
    "x.com": "X", "twitter.com": "X", "reddit.com": "Reddit", "wired.com": "WIRED",
}

_COMMON_CAPS = {"AI", "THE", "AND", "FOR"}


def source_name(urls: Iterable[str]) -> str:
    """A readable publisher name from the first usable source URL."""
    for url in urls or []:
        host = urlsplit(str(url)).netloc.lower().split(":")[0]
        if not host:
            continue
        host = host[4:] if host.startswith("www.") else host
        for known, nice in _KNOWN_SOURCES.items():
            if host == known or host.endswith("." + known):
                return nice
        parts = host.split(".")
        base = parts[-2] if len(parts) >= 2 else parts[0]
        return base.replace("-", " ").title()
    return ""


def extract_keywords(text: str, limit: int = 3) -> List[str]:
    """Distinctive words worth highlighting: names with digits, inner capitals, acronyms."""
    found: List[str] = []
    for raw in re.findall(r"[A-Za-z][A-Za-z0-9\-\.]{2,}", clean_display_text(text)):
        word = raw.strip(".-")
        if len(word) < 3 or word in found:
            continue
        has_digit = any(c.isdigit() for c in word)
        inner_cap = any(c.isupper() for c in word[1:]) and not word.isupper()
        acronym = word.isupper() and 3 <= len(word) <= 8 and word not in _COMMON_CAPS
        if has_digit or inner_cap or acronym:
            found.append(word)
    return found[:limit]


_STAT_RE = re.compile(
    r"(?<![A-Za-z0-9.])(?P<pre>[$€£₹])?(?P<num>\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)"
    r"(?P<unit>\s?(?:%|percent\b|x\b|×|trillion\b|billion\b|million\b|thousand\b|[BMK]\b))?",
    re.IGNORECASE,
)
_UNIT_WORDS = {"percent": "%", "trillion": "T", "billion": "B", "million": "M", "thousand": "K"}

_TRAIL = {"for", "the", "a", "an", "of", "to", "with", "in", "on", "and", "or", "by", "at",
          "from", "than", "while", "that", "is", "was", "are", "-", "as", "compared"}


def _tidy_label(words: List[str]) -> str:
    """Short caption for a stat; never ends mid-phrase on a dangling word or dash."""
    words = [w for w in words if w.strip("-\u2013\u2014")]
    while words and words[-1].strip(",.;:").lower() in _TRAIL:
        words.pop()
    return " ".join(words).strip(" ,-")

def extract_stats(text: str, limit: int = 2) -> List[Dict[str, Any]]:
    """Real figures quoted in the story text (percentages, money, multipliers, big counts).

    A bare number is not a statistic; it needs a unit, a currency sign or a
    percent sign. Years (1900-2100) are skipped.
    """
    stats: List[Dict[str, Any]] = []
    seen = set()
    text = clean_display_text(text)
    for m in _STAT_RE.finditer(text):
        pre, num, unit = m.group("pre") or "", m.group("num"), (m.group("unit") or "").strip()
        if not pre and not unit:
            continue
        if unit.lower() in ("x", "k", "m", "b") and not pre and "." not in num and len(num) > 3:
            continue
        value = float(num.replace(",", ""))
        if not pre and not unit and 1900 <= value <= 2100:
            continue
        suffix = _UNIT_WORDS.get(unit.lower(), unit.replace("×", "x"))
        display_key = (pre, num, suffix)
        if display_key in seen:
            continue
        seen.add(display_key)
        tail = re.split(r"[,;:!?]|\.(?:\s|$)", text[m.end():], maxsplit=1)[0].split()
        label = " ".join(tail[:4]).strip(",")
        stats.append({"value": value, "prefix": pre, "suffix": suffix,
                      "decimals": len(num.split(".")[1]) if "." in num else 0,
                      "label": _tidy_label(tail[:4])})
        if len(stats) >= limit:
            break
    return stats


def story_stats(story: Optional[Dict[str, Any]], narration: str = "") -> List[Dict[str, Any]]:
    """Statistics for one story from its own facts, never from elsewhere."""
    if not story:
        return extract_stats(narration, 1)
    facts = " ".join(str(f.get("statement", "")) for f in story.get("key_facts") or []
                     if isinstance(f, dict))
    return extract_stats(f"{story.get('summary', '')} {facts}", 1) or extract_stats(narration, 1)


def resolve_story_image(story: Optional[Dict[str, Any]], search_dirs: Iterable[Path]) -> Optional[Path]:
    """A user-supplied image for the story, or None (the animated graphic is used instead)."""
    if not story:
        return None
    candidates: List[Path] = []
    explicit = story.get("image_path")
    if explicit:
        candidates.append(Path(str(explicit)))
    story_id = str(story.get("story_id") or "")
    for folder in search_dirs:
        if not folder or not story_id:
            continue
        for suffix in IMAGE_SUFFIXES:
            candidates.append(Path(folder) / f"{story_id}{suffix}")
    for path in candidates:
        try:
            if path.is_file() and path.stat().st_size > 0:
                return path
        except OSError:
            continue
    return None


def build_scene_visuals(plan: Dict[str, Any], stories: List[Dict[str, Any]],
                        search_dirs: Iterable[Path]) -> List[Dict[str, Any]]:
    """One visual descriptor per scene (keywords, stats, source, optional image)."""
    by_id = {str(s.get("story_id")): s for s in stories if isinstance(s, dict)}
    search_dirs = list(search_dirs)
    visuals: List[Dict[str, Any]] = []
    for scene in plan["scenes"]:
        story = None
        story = by_id.get(str(scene.get("story_id") or ""))
        if story is None and scene.get("kind") == "story":
            story = _story_for_scene(scene, stories)
        headline = scene.get("headline") or scene.get("title") or ""
        is_story = scene.get("kind") == "story"
        visuals.append({
            "story_id": str(story.get("story_id")) if story else "",
            "keywords": extract_keywords(headline) if is_story else [],
            "stats": story_stats(story, scene.get("narration", "")) if is_story else [],
            "source": source_name(scene.get("source_urls") or []) if is_story else "",
            "category": clean_display_text((story or {}).get("category") or scene.get("topic") or ""),
            "image": resolve_story_image(story, search_dirs) if is_story else None,
        })
    return visuals


def _story_for_scene(scene: Dict[str, Any], stories: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    headline = clean_display_text(scene.get("headline") or "").casefold()
    for story in stories:
        if clean_display_text(story.get("headline") or "").casefold() == headline:
            return story
    return None