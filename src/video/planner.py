import re
from typing import Any, Dict, List, Tuple


_STOPWORDS = frozenset(
    "about above after again also another because been before being between "
    "both could does during each from have having into just like more most "
    "other over such than that their them then there these they this those "
    "through under very were what when where which while will with would "
    "your today researchers introduced new".split()
)

# Typographic characters that common fonts or speech engines handle badly.
_DISPLAY_CHAR_FIXES = {
    "\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2013": "-",
    "\u2014": " - ", "\u2212": "-", "\u00a0": " ", "\u202f": " ",
    "\u2009": " ", "\u200a": " ", "\u200b": "", "\ufeff": "",
}


_UNDRAWABLE = re.compile("[\U0001F000-\U0001FFFF\u2600-\u27BF\u2B00-\u2BFF\uFE00-\uFE0F\u200d]")


def clean_display_text(text: Any) -> str:
    """Normalise text for on-screen display or speech (never for the saved narration)."""
    value = str(text or "")
    for bad, good in _DISPLAY_CHAR_FIXES.items():
        value = value.replace(bad, good)
    # emoji / pictographs cannot be drawn by normal fonts (they show as empty boxes)
    value = _UNDRAWABLE.sub("", value)
    return re.sub(r"\s+", " ", value).strip()

def _shorten(text: Any, limit: int) -> str:
    value = clean_display_text(text)
    if len(value) <= limit:
        return value
    cut = value[:limit].rsplit(" ", 1)[0].rstrip(".,;:- ")
    return (cut or value[:limit]) + "..."


def _tokens(text: Any) -> set:
    return {
        word for word in re.findall(r"[a-z0-9]{4,}", str(text or "").lower())
        if word not in _STOPWORDS
    }


def _story_tokens(story: Dict[str, Any]) -> set:
    parts = [story.get("headline", ""), story.get("topic", ""), story.get("summary", "")]
    for fact in story.get("key_facts") or []:
        if isinstance(fact, dict):
            parts.append(fact.get("statement", ""))
    return _tokens(" ".join(str(part) for part in parts))


def _match_story_index(narration: str, story_tokens: List[set], previous: int) -> int:
    """Pick the story whose text the narration actually talks about.

    Falls back to the story already on screen when nothing matches clearly,
    so the picture never jumps to an unrelated story.
    """
    scene_tokens = _tokens(narration)
    best, best_score = previous, 0
    for index, tokens in enumerate(story_tokens):
        score = len(scene_tokens & tokens)
        if score > best_score:
            best, best_score = index, score
    return best if best_score >= 2 else previous


def apply_scene_durations(plan: Dict[str, Any], durations: List[float]) -> Dict[str, Any]:
    """Rewrite scene timings, subtitles and totals from per-scene durations (seconds)."""
    elapsed = 0.0
    srt_lines: List[str] = []
    subtitles: List[Dict[str, Any]] = []
    for index, (scene, duration) in enumerate(zip(plan["scenes"], durations), 1):
        start = round(elapsed, 3)
        end = round(elapsed + float(duration), 3)
        elapsed = end
        scene["duration_seconds"] = round(float(duration), 3)
        scene["start_seconds"] = start
        scene["end_seconds"] = end
        srt_lines.append(
            f"{index}\n{_format_srt_timestamp(start)} --> {_format_srt_timestamp(end)}\n"
            f"{scene['narration']}\n"
        )
        subtitles.append({
            "index": index, "start_seconds": start, "end_seconds": end,
            "text": scene["narration"],
        })
    plan["subtitles_srt"] = "\n".join(srt_lines).strip()
    plan["subtitles"] = subtitles
    plan["actual_duration_seconds"] = round(elapsed, 3)
    return plan


def _split_into_sentences(text: str) -> List[str]:
    """Split text into exact sentences while preserving sentence words."""
    cleaned = text.strip()
    if not cleaned:
        return []
    # Split on sentence boundaries (. ! ?) followed by whitespace
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", cleaned) if s.strip()]
    return sentences or [cleaned]


def _format_srt_timestamp(seconds: float) -> str:
    """Format seconds into SRT timestamp format: HH:MM:SS,mmm."""
    total_ms = int(round(seconds * 1000))
    hours = total_ms // 3600000
    minutes = (total_ms % 3600000) // 60000
    secs = (total_ms % 60000) // 1000
    ms = total_ms % 1000
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def _assign_durations(word_counts: List[int], target_duration: int) -> List[int]:
    """Proportionally allocate duration across scenes ensuring exact sum."""
    total_words = sum(word_counts)
    if total_words == 0 or not word_counts:
        return [target_duration]

    durations = [
        max(2, int(round((count / total_words) * target_duration)))
        for count in word_counts
    ]
    diff = target_duration - sum(durations)
    durations[-1] += diff
    if durations[-1] < 1:
        durations[-1] = 1
    return durations


def plan_video_scenes(
    programme: Dict[str, Any],
    programme_type: str = "one_minute"
) -> Dict[str, Any]:
    """
    Divide the exact narration from daily_programme.json into synchronized video scenes.

    output/daily_programme.json is the single source of truth.
    Never alters or rewrites the original narration.
    """
    if not isinstance(programme, dict):
        raise ValueError("Programme must be a dictionary.")

    programme_date = programme.get("programme_date", "")
    programmes = programme.get("programmes", {})
    stories = programme.get("stories", [])

    if programme_type not in programmes:
        raise ValueError(
            f"Programme type '{programme_type}' not found in daily programme. "
            f"Available types: {list(programmes.keys())}"
        )

    selected = programmes[programme_type]
    narration = str(selected.get("narration", "")).strip()
    if not narration:
        raise ValueError(f"No narration found for '{programme_type}'.")

    target_duration = int(selected.get(
        "duration_seconds",
        60 if programme_type == "one_minute" else 180
    ))
    title = str(selected.get("title", f"Daily AI Intelligence - {programme_date}"))

    sentences = _split_into_sentences(narration)
    n = len(sentences)

    # Group sentences into cohesive scenes:
    # Scene 1: Introduction (1-2 sentences)
    # Scenes 2..k: Body / Stories
    # Scene Last: Outro (1-2 sentences if n >= 4)
    story_token_sets = [_story_tokens(story) for story in stories]
    max_sentences_per_scene = 2

    scene_sentence_groups: List[Tuple[str, List[str]]] = []
    if n <= 3:
        for idx, sentence in enumerate(sentences, 1):
            scene_sentence_groups.append((f"Scene {idx}", [sentence]))
    else:
        # Opening/closing sentences that talk about no story become the intro and
        # outro. If the narration starts straight into a story there is no intro,
        # so that story's slide is shown instead of a title card.
        matches = [
            _match_story_index(sentence, story_token_sets, None) if stories else None
            for sentence in sentences
        ]
        lead = 0
        while lead < n and matches[lead] is None:
            lead += 1
        tail = 0
        while tail < n - lead and matches[-1 - tail] is None:
            tail += 1
        if not stories or lead + tail >= n:
            # Nothing recognisable: fall back to a fixed intro and outro.
            lead = tail = (1 if n <= 6 else 2)
            matches = [None] * n
        lead, tail = min(lead, 3), min(tail, 3)

        mid_matches = matches[lead:n - tail]
        mid = sentences[lead:n - tail]
        # A sentence that names no story belongs to the story before it
        # (or the next one, at the start).
        filled: List[int] = []
        previous_story = next((m for m in mid_matches if m is not None), 0)
        for match in mid_matches:
            previous_story = previous_story if match is None else match
            filled.append(previous_story)

        if lead:
            scene_sentence_groups.append(("Introduction", sentences[:lead]))
        runs: List[List[Any]] = []
        for sentence, story_index in zip(mid, filled):
            if runs and runs[-1][0] == story_index:
                runs[-1][1].append(sentence)
            else:
                runs.append([story_index, [sentence]])
        for _, run in runs:
            for i in range(0, len(run), max_sentences_per_scene):
                scene_sentence_groups.append((
                    f"Story {len(scene_sentence_groups) + 1}",
                    run[i:i + max_sentences_per_scene],
                ))
        if tail:
            scene_sentence_groups.append(("Closing & Sign-Off", sentences[n - tail:]))

    # Match stories to source references and on-screen text
    primary_urls = selected.get("source_urls", [])
    if not primary_urls and stories:
        primary_urls = stories[0].get("source_urls", [])

    scene_word_counts = [
        sum(len(s.split()) for s in s_list)
        for _, s_list in scene_sentence_groups
    ]
    durations = _assign_durations(scene_word_counts, target_duration)

    last_story_idx = 0
    scenes: List[Dict[str, Any]] = []
    subtitles: List[Dict[str, Any]] = []
    srt_lines: List[str] = []

    current_time = 0.0

    for idx, ((label, s_list), dur) in enumerate(zip(scene_sentence_groups, durations), 1):
        scene_narration = " ".join(s_list)
        start_time = current_time
        end_time = round(start_time + dur, 2)
        current_time = end_time

        # Associate story metadata by what the narration is actually about
        if stories:
            story_idx = _match_story_index(scene_narration, story_token_sets, last_story_idx)
            last_story_idx = story_idx
            matched_story = stories[story_idx]
        else:
            matched_story = {}
        story_urls = matched_story.get("source_urls", primary_urls)
        story_headline = clean_display_text(matched_story.get("headline", title))
        story_topic = clean_display_text(matched_story.get("topic", ""))

        small = n <= 3
        is_intro = label == "Introduction" or (small and idx == 1)
        is_outro = label == "Closing & Sign-Off" or (small and idx == len(scene_sentence_groups))
        if is_intro:
            kind = "intro"
            scene_title = title
            visual = "Anchor graphic with futuristic AI news backdrop and daily date badge."
            key_points = []
            on_screen = f"AI DAILY INTELLIGENCE\n{programme_date}\n{title}"
        elif is_outro:
            kind = "outro"
            scene_title = "That's your AI briefing for today"
            visual = "Summary graphic with verified source checklist."
            key_points = ["Full report and papers are in today's PDF"]
            on_screen = "FULL REPORT & PAPERS AVAILABLE IN DAILY PDF"
        else:
            kind = "story"
            scene_title = story_topic or story_headline
            visual = f"Diagram illustrating: {_shorten(matched_story.get('why_it_matters', scene_narration), 70)}"
            key_points = [
                _shorten(fact.get("statement", ""), 150)
                for fact in (matched_story.get("key_facts") or [])[:3]
                if isinstance(fact, dict) and fact.get("statement")
            ]
            if not key_points and matched_story.get("why_it_matters"):
                key_points = [_shorten(matched_story["why_it_matters"], 150)]
            on_screen = "\n".join([story_headline] + key_points)

        scene_obj = {
            "scene_number": idx,
            "kind": kind,
            "title": scene_title,
            "headline": story_headline if kind == "story" else scene_title,
            "topic": story_topic,
            "key_points": key_points,
            "label": label,
            "narration": scene_narration,
            "duration_seconds": dur,
            "start_seconds": start_time,
            "end_seconds": end_time,
            "visual_suggestion": visual,
            "on_screen_text": on_screen,
            "source_urls": story_urls,
            "story_id": str(matched_story.get("story_id", "")) if kind == "story" else "",
        }
        scenes.append(scene_obj)

        # Build Subtitle entry for scene
        srt_idx = idx
        srt_start = _format_srt_timestamp(start_time)
        srt_end = _format_srt_timestamp(end_time)
        srt_lines.append(f"{srt_idx}\n{srt_start} --> {srt_end}\n{scene_narration}\n")
        subtitles.append({
            "index": srt_idx,
            "start_seconds": start_time,
            "end_seconds": end_time,
            "text": scene_narration
        })

    # Validate that exact narration is preserved character-for-character
    reconstructed = " ".join(scene["narration"] for scene in scenes).strip()
    if " ".join(reconstructed.split()) != " ".join(narration.split()):
        raise AssertionError("Scene planning altered the original narration text!")

    return {
        "programme_date": programme_date,
        "programme_type": programme_type,
        "title": title,
        "target_duration_seconds": target_duration,
        "actual_duration_seconds": sum(s["duration_seconds"] for s in scenes),
        "narration": narration,
        "word_count": len(narration.split()),
        "scene_count": len(scenes),
        "scenes": scenes,
        "subtitles_srt": "\n".join(srt_lines).strip(),
        "subtitles": subtitles,
        "stories": stories
    }