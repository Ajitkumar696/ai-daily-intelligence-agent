import json
import os
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlsplit

from PIL import Image, ImageDraw, ImageFont

from .planner import apply_scene_durations, clean_display_text, plan_video_scenes


DEFAULT_VIDEO_DIR = Path(__file__).resolve().parents[2] / "output" / "videos"

WIDTH, HEIGHT = 1920, 1080
MARGIN = 100
# Quiet time after each scene's speech, so scenes do not run into each other.
SCENE_PAD_SECONDS = 0.5

# Palette
BG = (15, 23, 42)            # Slate 900
PANEL = (30, 41, 59)         # Slate 800
CAPTION_BG = (2, 6, 23)
BLUE = (56, 189, 248)        # Sky 400
TEAL = (45, 212, 191)        # Teal 400
WHITE = (248, 250, 252)
MUTED = (148, 163, 184)
LINE = (51, 65, 85)

_REGULAR_FONTS = (
    "segoeui.ttf", "arial.ttf", "calibri.ttf", "DejaVuSans.ttf",
    "LiberationSans-Regular.ttf", "NotoSans-Regular.ttf", "Helvetica.ttc",
)
_BOLD_FONTS = (
    "segoeuib.ttf", "arialbd.ttf", "calibrib.ttf", "DejaVuSans-Bold.ttf",
    "LiberationSans-Bold.ttf", "NotoSans-Bold.ttf", "Helvetica.ttc",
)
_FONT_DIRS = (
    None,  # let Pillow search the platform font folders by file name
    "C:/Windows/Fonts",
    "/usr/share/fonts/truetype/dejavu",
    "/usr/share/fonts/truetype/liberation",
    "/usr/share/fonts/truetype/noto",
    "/Library/Fonts",
    "/System/Library/Fonts",
)


@lru_cache(maxsize=64)
def _font(size: int, bold: bool = False):
    """A real TrueType font at ``size`` px (the default Pillow font is ~11 px)."""
    for name in (_BOLD_FONTS if bold else _REGULAR_FONTS):
        for folder in _FONT_DIRS:
            path = name if folder is None else os.path.join(folder, name)
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    try:
        return ImageFont.load_default(size)
    except TypeError:  # very old Pillow
        return ImageFont.load_default()


def _size_of(font) -> int:
    return int(getattr(font, "size", 20))


def _wrap(draw, text: str, font, max_width: int) -> List[str]:
    lines: List[str] = []
    line = ""
    for word in text.split():
        trial = f"{line} {word}".strip()
        if not line or draw.textlength(trial, font=font) <= max_width:
            line = trial
        else:
            lines.append(line)
            line = word
    if line:
        lines.append(line)
    return lines


def _fit(draw, text: str, bold: bool, sizes: Tuple[int, ...], max_width: int,
         max_lines: int):
    """Largest size from ``sizes`` whose wrapped text fits ``max_lines``."""
    text = clean_display_text(text)
    for size in sizes:
        font = _font(size, bold)
        lines = _wrap(draw, text, font, max_width)
        if len(lines) <= max_lines:
            return font, lines
    font = _font(sizes[-1], bold)
    lines = _wrap(draw, text, font, max_width)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = lines[-1].rstrip(".,;:- ") + "..."
    return font, lines


def _draw_lines(draw, lines: List[str], font, x: int, y: int, fill,
                spacing: float = 1.28) -> int:
    step = int(_size_of(font) * spacing)
    for line in lines:
        draw.text((x, y), line, fill=fill, font=font)
        y += step
    return y


def _pill(draw, x: int, y: int, text: str, color) -> int:
    font = _font(26, True)
    label = text.upper()
    width = int(draw.textlength(label, font=font)) + 44
    draw.rounded_rectangle([(x, y), (x + width, y + 50)], radius=25,
                           outline=color, width=2)
    draw.text((x + 22, y + 9), label, fill=color, font=font)
    return y + 50


def _source_label(scene: Dict[str, Any]) -> str:
    urls = scene.get("source_urls") or []
    if not urls:
        return ""
    host = urlsplit(str(urls[0])).netloc.lower()
    host = host[4:] if host.startswith("www.") else host
    return f"Source: {host}" if host else ""


def _draw_caption(draw, narration: str) -> None:
    top, bottom = 790, 1000
    draw.rounded_rectangle([(MARGIN, top), (WIDTH - MARGIN, bottom)], radius=18,
                           fill=CAPTION_BG, outline=BLUE, width=2)
    inner_w = WIDTH - 2 * MARGIN - 80
    inner_h = bottom - top - 44
    text = clean_display_text(narration)
    chosen = None
    for size in (44, 40, 36, 32, 28, 25):
        font = _font(size)
        lines = _wrap(draw, text, font, inner_w)
        if len(lines) * int(size * 1.3) <= inner_h:
            chosen = (font, lines)
            break
    if chosen is None:
        font = _font(25)
        max_lines = max(1, inner_h // int(25 * 1.3))
        lines = _wrap(draw, text, font, inner_w)[:max_lines]
        lines[-1] = lines[-1].rstrip(".,;:- ") + "..."
        chosen = (font, lines)
    font, lines = chosen
    block_h = len(lines) * int(_size_of(font) * 1.3)
    y = top + (bottom - top - block_h) // 2
    _draw_lines(draw, lines, font, MARGIN + 40, y, WHITE, spacing=1.3)


def _draw_progress(draw, index: int, total: int) -> None:
    gap = 10
    seg_w = (WIDTH - 2 * MARGIN - gap * (total - 1)) / max(total, 1)
    for i in range(total):
        x0 = MARGIN + i * (seg_w + gap)
        color = TEAL if i < index else LINE
        if i == index - 1:
            color = BLUE
        draw.rounded_rectangle([(x0, 1054), (x0 + seg_w, 1062)], radius=4, fill=color)


def render_scene_frame(
    scene: Dict[str, Any],
    programme_meta: Dict[str, Any],
    output_path: Path
) -> Path:
    """Render a readable 1920x1080 scene card: headline, key points, live caption."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    img = Image.new("RGB", (WIDTH, HEIGHT), color=BG)
    draw = ImageDraw.Draw(img)

    # Header
    draw.text((MARGIN, 44), "AI DAILY INTELLIGENCE", fill=BLUE, font=_font(32, True))
    date_text = str(programme_meta.get("programme_date") or "")
    date_font = _font(30)
    draw.text((WIDTH - MARGIN - draw.textlength(date_text, font=date_font), 46),
              date_text, fill=MUTED, font=date_font)
    draw.line([(MARGIN, 104), (WIDTH - MARGIN, 104)], fill=LINE, width=2)

    kind = scene.get("kind", "story")
    headline = scene.get("headline") or scene.get("title") or ""
    text_w = WIDTH - 2 * MARGIN
    y = 150

    if kind == "intro":
        y = _pill(draw, MARGIN, y, "Daily briefing", TEAL) + 40
        font, lines = _fit(draw, headline, True, (88, 80, 72, 64), text_w, 4)
        y = _draw_lines(draw, lines, font, MARGIN, y, WHITE, spacing=1.2) + 24
        draw.text((MARGIN, y), "Your daily AI briefing", fill=MUTED, font=_font(40))
    elif kind == "outro":
        y = _pill(draw, MARGIN, y, "Wrap-up", TEAL) + 40
        font, lines = _fit(draw, headline, True, (84, 76, 68), text_w, 3)
        y = _draw_lines(draw, lines, font, MARGIN, y, WHITE, spacing=1.2) + 28
        for point in scene.get("key_points", [])[:3]:
            y = _bullet(draw, point, y)
    else:
        topic = clean_display_text(scene.get("topic") or "")
        if topic and topic.casefold() != clean_display_text(headline).casefold():
            y = _pill(draw, MARGIN, y, topic[:40], TEAL) + 36
        bar_top = y
        font, lines = _fit(draw, headline, True, (68, 62, 56, 50), text_w, 3)
        y = _draw_lines(draw, lines, font, MARGIN, y, WHITE, spacing=1.2) + 30
        for point in scene.get("key_points", [])[:3]:
            if y > 700:
                break
            y = _bullet(draw, point, y)
        draw.rectangle([(MARGIN - 28, bar_top), (MARGIN - 20, min(y, 760))], fill=BLUE)

    _draw_caption(draw, scene.get("narration", ""))

    source = _source_label(scene) if kind == "story" else ""
    if source:
        draw.text((MARGIN, 1010), source, fill=MUTED, font=_font(24))
    total = int(programme_meta.get("scene_count") or len(programme_meta.get("scenes", [])) or 1)
    _draw_progress(draw, int(scene.get("scene_number", 1)), total)

    img.save(output_path, format="PNG")
    return output_path


def _bullet(draw, text: str, y: int) -> int:
    font, lines = _fit(draw, text, False, (40, 36, 32), WIDTH - 2 * MARGIN - 50, 3)
    cy = y + int(_size_of(font) * 0.62)
    draw.ellipse([(MARGIN, cy - 8), (MARGIN + 16, cy + 8)], fill=BLUE)
    return _draw_lines(draw, lines, font, MARGIN + 50, y, (226, 232, 240), 1.28) + 18


def generate_daily_video(
    programme: Dict[str, Any],
    programme_type: str = "one_minute",
    output_dir: Optional[Path] = None,
    render_mp4_if_possible: bool = True,
    config=None,
    providers=None,
    log=print,
) -> Dict[str, Any]:
    """Generate the animated news video package from daily_programme.json.

    daily_programme.json stays the single source of truth and the narration is
    used exactly as written. Stages live in ``pipeline.run_pipeline``. Raises
    ``TTSError`` (real cause) when no narration audio can be produced and
    ``RenderError`` when the rendered MP4 fails validation.
    """
    from .pipeline import run_pipeline
    return run_pipeline(programme, programme_type, output_dir, render_mp4_if_possible,
                        config=config, providers=providers, log=log)