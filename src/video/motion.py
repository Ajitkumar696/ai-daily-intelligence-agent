"""Motion-graphics frame renderer (Pillow). ``MotionRenderer.render_frame(t)``
returns the 1920x1080 RGB frame for time ``t`` seconds of the video.

Look: dark navy gradient, neon cyan + purple accents, white type, drifting
particles, word-by-word headlines, news lower third, karaoke subtitles,
animated transitions and a progress timeline. Everything drawn comes from the
programme data; if a story has no supplied image an animated graphic is used.
"""
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFilter, ImageOps

from .config import HEIGHT, WIDTH
from .planner import clean_display_text
from .renderer import _font, _wrap
from .timing import Cue, WordTiming, active_word_index, cue_at

NAVY_TOP = (4, 7, 20)
NAVY_BOTTOM = (13, 18, 46)
CYAN = (34, 224, 255)
PURPLE = (168, 85, 247)
WHITE = (246, 249, 255)
SOFT = (205, 215, 235)
MUTED = (140, 152, 180)
LINE = (38, 48, 84)
GLASS = (11, 17, 42)

MARGIN = 100
PANEL_W, PANEL_H = 800, 480
SUB_TOP, SUB_BOTTOM = 812, 990
TIMELINE_Y = 1040
ENTER, EXIT = 0.55, 0.35


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if x < lo else hi if x > hi else x


def ease_out(x: float) -> float:
    x = _clamp(x)
    return 1 - (1 - x) ** 3


def _scale(color: Tuple[int, int, int], k: float) -> Tuple[int, int, int]:
    return tuple(int(_clamp(c * k, 0, 255)) for c in color)


class Sprite:
    """Pre-rendered text: a solid colour image plus an alpha mask."""

    def __init__(self, text: str, font, color):
        l, t, r, b = font.getbbox(text)
        w, h = max(int(r) + 6, 1), max(int(b) + 6, 1)
        mask = Image.new("L", (w, h), 0)
        ImageDraw.Draw(mask).text((0, 0), text, fill=255, font=font)
        self.mask = mask
        self.img = Image.new("RGB", (w, h), color)
        self.w, self.h = w, h
        self._lut_cache: Dict[int, Image.Image] = {}

    def paste(self, base: Image.Image, x: int, y: int, alpha: float = 1.0) -> None:
        if alpha <= 0.01:
            return
        if alpha >= 0.99:
            mask = self.mask
        else:
            level = int(alpha * 20)  # quantised so masks can be cached
            mask = self._lut_cache.get(level)
            if mask is None:
                f = level / 20.0
                mask = self.mask.point([int(v * f) for v in range(256)])
                self._lut_cache[level] = mask
        base.paste(self.img, (int(x), int(y)), mask)


def _rounded_mask(size: Tuple[int, int], radius: int, value: int = 255) -> Image.Image:
    mask = Image.new("L", size, 0)
    ImageDraw.Draw(mask).rounded_rectangle([(0, 0), (size[0] - 1, size[1] - 1)],
                                           radius=radius, fill=value)
    return mask


def _blend_paste(base: Image.Image, layer: Image.Image, pos: Tuple[int, int],
                 mask: Image.Image, alpha: float) -> None:
    if alpha <= 0.01:
        return
    box = (int(pos[0]), int(pos[1]), int(pos[0]) + layer.width, int(pos[1]) + layer.height)
    if alpha < 0.99:
        region = base.crop(box)
        layer = Image.blend(region, layer, alpha)
    base.paste(layer, box[:2], mask)


class MotionRenderer:
    def __init__(self, plan: Dict[str, Any], visuals: List[Dict[str, Any]],
                 segments: List[Dict[str, Any]], total_seconds: float,
                 voice_credit: str = ""):
        self.plan = plan
        self.visuals = visuals
        self.segments = segments
        self.total = total_seconds
        self.voice_credit = voice_credit
        self.date = str(plan.get("programme_date") or "")
        self._sprites: Dict[Any, Any] = {}
        self._images: Dict[str, Optional[Image.Image]] = {}
        self._cue_layout: Dict[int, Any] = {}
        self._rng_cache: Dict[str, Any] = {}
        self._build_static()

    # ------------------------------------------------------------ static art
    def _gradient_bg(self) -> Image.Image:
        grad = Image.linear_gradient("L").resize((WIDTH, HEIGHT))
        bg = Image.composite(Image.new("RGB", (WIDTH, HEIGHT), NAVY_BOTTOM),
                             Image.new("RGB", (WIDTH, HEIGHT), NAVY_TOP), grad)
        glow = ImageOps.invert(Image.radial_gradient("L")).resize((1500, 1500))
        for color, pos, strength in ((CYAN, (-520, -620), 0.20), (PURPLE, (900, 380), 0.22)):
            layer = Image.new("RGB", glow.size, color)
            # fade to exactly zero inside the square so no hard edge shows
            lut = [int(min(max((v - 128) * 2, 0), 255) * strength) for v in range(256)]
            bg.paste(layer, pos, glow.point(lut))
        draw = ImageDraw.Draw(bg)
        for x in range(0, WIDTH, 120):  # faint tech grid
            draw.line([(x, 0), (x, HEIGHT)], fill=(10, 16, 40), width=1)
        for y in range(0, HEIGHT, 120):
            draw.line([(0, y), (WIDTH, y)], fill=(10, 16, 40), width=1)
        return bg

    def _build_static(self) -> None:
        self.bg_plain = self._gradient_bg()
        self.bg_header = self.bg_plain.copy()
        d = ImageDraw.Draw(self.bg_header)
        d.ellipse([(MARGIN, 38), (MARGIN + 34, 72)], outline=CYAN, width=4)
        d.ellipse([(MARGIN + 11, 49), (MARGIN + 23, 61)], fill=PURPLE)
        d.text((MARGIN + 50, 36), "AI DAILY", fill=WHITE, font=_font(32, True))
        d.text((MARGIN + 50 + d.textlength("AI DAILY ", font=_font(32, True)), 36),
               "INTELLIGENCE", fill=CYAN, font=_font(32, True))
        f = _font(28)
        tag = f"DAILY BRIEFING  |  {self.date}"
        d.text((WIDTH - MARGIN - d.textlength(tag, font=f), 40), tag, fill=MUTED, font=f)
        d.line([(MARGIN, 96), (WIDTH - MARGIN, 96)], fill=LINE, width=2)
        d.line([(MARGIN, 96), (MARGIN + 220, 96)], fill=CYAN, width=3)

        rng = random.Random(7)
        self.particles = [(rng.uniform(0, WIDTH), rng.uniform(0, HEIGHT), rng.uniform(8, 28),
                           rng.uniform(1.5, 3.6), rng.uniform(0, 6.28), rng.random() < 0.55)
                          for _ in range(46)]
        self.panel_mask = _rounded_mask((PANEL_W, PANEL_H), 28)
        self.panel_bg = self._panel_background()
        self.plate = (Image.new("RGB", (WIDTH - 2 * MARGIN, SUB_BOTTOM - SUB_TOP), (2, 4, 14)),
                      _rounded_mask((WIDTH - 2 * MARGIN, SUB_BOTTOM - SUB_TOP), 22, 168))

    def _panel_background(self) -> Image.Image:
        grad = Image.linear_gradient("L").resize((PANEL_W, PANEL_H))
        img = Image.composite(Image.new("RGB", (PANEL_W, PANEL_H), (16, 22, 56)),
                              Image.new("RGB", (PANEL_W, PANEL_H), GLASS), grad)
        d = ImageDraw.Draw(img)
        for x in range(30, PANEL_W, 40):
            for y in range(30, PANEL_H, 40):
                d.point((x, y), fill=(44, 58, 104))
        d.rounded_rectangle([(1, 1), (PANEL_W - 2, PANEL_H - 2)], radius=28,
                            outline=(30, 120, 160), width=2)
        return img

    # ----------------------------------------------------------- dispatching
    def segment_at(self, t: float) -> Dict[str, Any]:
        for seg in self.segments:
            if t < seg["end"]:
                return seg
        return self.segments[-1]

    def render_frame(self, t: float) -> Image.Image:
        seg = self.segment_at(t)
        base = (self.bg_plain if seg["type"] != "scene" else self.bg_header).copy()
        self._particles(base, t)
        if seg["type"] == "bumper":
            self._bumper(base, t - seg["start"], seg["end"] - seg["start"])
        elif seg["type"] == "closing":
            self._closing(base, t - seg["start"], seg["end"] - seg["start"])
        else:
            self._scene(base, seg, t)
            self._timeline(base, t)
        return base

    # -------------------------------------------------------------- particles
    def _particles(self, base: Image.Image, t: float) -> None:
        d = ImageDraw.Draw(base)
        for x0, y0, speed, size, phase, is_cyan in self.particles:
            y = (y0 - speed * t) % HEIGHT
            x = x0 + 14 * math.sin(t * 0.45 + phase)
            glow = 0.30 + 0.22 * math.sin(t * 1.3 + phase)
            col = _scale(CYAN if is_cyan else PURPLE, glow)
            d.ellipse([(x - size, y - size), (x + size, y + size)], fill=col)

    # ----------------------------------------------------------------- helpers
    def _sprite(self, key, text: str, font, color) -> Sprite:
        sprite = self._sprites.get(key)
        if sprite is None:
            sprite = self._sprites[key] = Sprite(text, font, color)
        return sprite

    def _layout_headline(self, key, text: str, keywords: Sequence[str], width: int,
                         sizes=(68, 62, 56, 50, 44, 40), max_lines=4):
        cached = self._sprites.get(("hl", key))
        if cached:
            return cached
        draw = ImageDraw.Draw(Image.new("RGB", (4, 4)))
        text = clean_display_text(text)
        words = text.split()
        chosen = None
        for size in sizes:
            font = _font(size, True)
            lines = _wrap(draw, text, font, width)
            if len(lines) <= max_lines:
                chosen = (size, font, lines)
                break
        if chosen is None:
            size = sizes[-1]
            font = _font(size, True)
            lines = _wrap(draw, text, font, width)[:max_lines]
            lines[-1] = lines[-1].rstrip(".,;:- ") + "..."
            chosen = (size, font, lines)
        size, font, lines = chosen
        kw = {k.casefold() for k in keywords}
        space = draw.textlength(" ", font=font)
        items = []
        for li, line in enumerate(lines):
            x = 0.0
            for word in line.split():
                core = word.strip(".,;:!?\"'()")
                color = CYAN if core.casefold() in kw else WHITE
                sprite = Sprite(word, font, color)
                items.append((sprite, x, int(li * size * 1.22)))
                x += draw.textlength(word, font=font) + space
        result = (items, int(len(lines) * size * 1.22))
        self._sprites[("hl", key)] = result
        return result

    def _draw_headline(self, base, key, text, keywords, x, y, width, lt, alpha, delay=0.25):
        items, height = self._layout_headline(key, text, keywords, width)
        for i, (sprite, dx, dy) in enumerate(items):
            p = ease_out((lt - delay - i * 0.07) / 0.45)
            sprite.paste(base, x + dx + (1 - p) * 26, y + dy, p * alpha)
        return height

    def _pill(self, base, key, text, x, y, color, alpha):
        font = _font(24, True)
        label = text.upper()
        if len(label) > 30:  # cut at a word boundary, never mid-phrase
            label = label[:30].rsplit(" ", 1)[0].rstrip(" ,-:") + "..."
        sprite = self._sprite(("pill", key), label, font, color)
        w = sprite.w + 40
        layer = Image.new("RGB", (w, 46), (14, 22, 52))
        d = ImageDraw.Draw(layer)
        d.rounded_rectangle([(0, 0), (w - 1, 45)], radius=23, outline=color, width=2)
        mask = _rounded_mask((w, 46), 23)
        _blend_paste(base, layer, (x, y), mask, alpha)
        sprite.paste(base, x + 20, y + 8, alpha)
        return w

    # ------------------------------------------------------------------ scene
    def _scene(self, base: Image.Image, seg: Dict[str, Any], t: float) -> None:
        idx = seg["index"]
        scene = self.plan["scenes"][idx]
        vis = self.visuals[idx]
        lt, dur = t - seg["start"], seg["end"] - seg["start"]
        alpha = min(ease_out(lt / ENTER), _clamp((dur - lt) / EXIT))
        kind = scene.get("kind", "story")
        variant = idx % 2  # alternate layouts: text-left / text-right
        tx, tw = (MARGIN, 880) if variant == 0 else (960, 860)
        px = 1000 + 20 if variant == 0 else MARGIN

        headline = scene.get("headline") or scene.get("title") or ""
        if kind == "intro":
            kicker, kcolor = "Today's briefing", CYAN
        elif kind == "outro":
            kicker, kcolor = "Wrap-up", PURPLE
        else:
            kicker = vis.get("category") or "AI news"
            kcolor = CYAN if idx % 2 == 0 else PURPLE
        self._pill(base, ("k", idx), kicker, tx, 150, kcolor, alpha)
        h = self._draw_headline(base, idx, headline, vis.get("keywords", []), tx, 222, tw, lt, alpha)
        bar_w = int(300 * ease_out((lt - 0.5) / 0.7))
        if bar_w > 0:
            ImageDraw.Draw(base).rectangle([(tx, 222 + h + 14), (tx + bar_w, 222 + h + 19)], fill=kcolor)

        self._panel(base, idx, scene, vis, px, 160, lt, dur, alpha, variant)
        self._lower_third(base, idx, vis, lt, dur)
        self._subtitles(base, seg, t)
        self._wipe(base, lt)

    # ------------------------------------------------------------ visual panel
    def _load_image(self, path: Path) -> Optional[Image.Image]:
        key = str(path)
        if key not in self._images:
            try:
                img = Image.open(path).convert("RGB")
                img.thumbnail((2200, 2200))
                self._images[key] = img
            except Exception:  # unreadable/corrupt: fall back to the graphic
                self._images[key] = None
        return self._images[key]

    def _panel(self, base, idx, scene, vis, px, py, lt, dur, alpha, variant):
        slide = (1 - ease_out(lt / (ENTER + 0.2))) * (70 if variant == 0 else -70)
        panel = self.panel_bg.copy()
        image = self._load_image(vis["image"]) if vis.get("image") else None
        if image is not None:
            self._ken_burns(panel, image, lt / max(dur, 0.1), idx)
        else:
            self._network(panel, vis.get("story_id") or scene.get("kind", "x"), lt, scene.get("kind"))
        stats = vis.get("stats") or []
        if stats and scene.get("kind") == "story":
            self._stat_card(panel, stats[0], lt)
        _blend_paste(base, panel, (int(px + slide), py), self.panel_mask, alpha)

    def _ken_burns(self, panel, image, u, idx):
        iw, ih = image.size
        s = max(PANEL_W / iw, PANEL_H / ih)
        zoom = 1.0 + 0.14 * _clamp(u)
        cw, ch = PANEL_W / (s * zoom), PANEL_H / (s * zoom)
        direction = -1 if idx % 2 else 1
        cx = iw / 2 + direction * (iw - cw) / 2 * (_clamp(u) * 2 - 1) * 0.7
        cy = ih / 2 + (ih - ch) / 2 * (0.5 - _clamp(u)) * 0.5
        left = _clamp(cx - cw / 2, 0, iw - cw)
        top = _clamp(cy - ch / 2, 0, ih - ch)
        frame = image.resize((PANEL_W, PANEL_H), Image.BILINEAR, box=(left, top, left + cw, top + ch))
        panel.paste(frame, (0, 0))
        ImageDraw.Draw(panel).rounded_rectangle([(1, 1), (PANEL_W - 2, PANEL_H - 2)], radius=28,
                                                outline=CYAN, width=3)

    def _network(self, panel, seed, lt, kind):
        cache = self._rng_cache.get(seed)
        if cache is None:
            rng = random.Random(str(seed))
            nodes = [(rng.uniform(70, PANEL_W - 70), rng.uniform(60, PANEL_H - 60),
                      rng.uniform(4, 9), rng.uniform(0, 6.28), rng.random() < 0.5)
                     for _ in range(17)]
            cache = self._rng_cache[seed] = nodes
        nodes = cache
        d = ImageDraw.Draw(panel)
        pos = [(x + 14 * math.sin(lt * 0.7 + ph), y + 11 * math.cos(lt * 0.6 + ph))
               for x, y, _, ph, _ in nodes]
        for i in range(len(pos)):
            for j in range(i + 1, len(pos)):
                dist = math.dist(pos[i], pos[j])
                if dist < 210:
                    k = 1 - dist / 210
                    d.line([pos[i], pos[j]], fill=_scale(CYAN, 0.12 + 0.45 * k), width=2 if k > 0.5 else 1)
                    if (i * 7 + j) % 5 == 0:  # data pulse travelling along the link
                        f = (lt * 0.5 + (i + j) * 0.13) % 1
                        px_ = pos[i][0] + (pos[j][0] - pos[i][0]) * f
                        py_ = pos[i][1] + (pos[j][1] - pos[i][1]) * f
                        d.ellipse([(px_ - 4, py_ - 4), (px_ + 4, py_ + 4)], fill=WHITE)
        for (x, y), (_, _, r, ph, is_cyan) in zip(pos, nodes):
            pulse = 1 + 0.35 * math.sin(lt * 2.2 + ph)
            col = CYAN if is_cyan else PURPLE
            d.ellipse([(x - r * pulse * 2, y - r * pulse * 2), (x + r * pulse * 2, y + r * pulse * 2)],
                      outline=_scale(col, 0.4), width=1)
            d.ellipse([(x - r, y - r), (x + r, y + r)], fill=col)
        cx, cy = PANEL_W / 2, PANEL_H / 2
        for radius, speed, col in ((120, 38, CYAN), (170, -26, PURPLE)):
            d.ellipse([(cx - radius, cy - radius), (cx + radius, cy + radius)],
                      outline=_scale(col, 0.25), width=2)
            a0 = lt * speed
            d.arc([(cx - radius, cy - radius), (cx + radius, cy + radius)], a0, a0 + 70, fill=col, width=5)

    def _stat_card(self, panel, stat, lt):
        p = ease_out((lt - 0.7) / 0.5)
        if p <= 0:
            return
        count = ease_out((lt - 0.8) / 1.5)
        value = stat["value"] * count
        number = f"{value:,.{stat['decimals']}f}" if stat["decimals"] else f"{value:,.0f}"
        text = f"{stat['prefix']}{number}{stat['suffix']}"
        font, small = _font(112, True), _font(30)
        d0 = ImageDraw.Draw(panel)
        w = int(max(d0.textlength(text, font=font), d0.textlength(stat["label"], font=small))) + 80
        h = 210 if stat["label"] else 160
        x0, y0 = (PANEL_W - w) // 2, int((PANEL_H - h) / 2 + (1 - p) * 30)
        card = Image.new("RGB", (w, h), (8, 12, 32))
        cd = ImageDraw.Draw(card)
        cd.rounded_rectangle([(0, 0), (w - 1, h - 1)], radius=22, outline=CYAN, width=3)
        cd.text(((w - cd.textlength(text, font=font)) / 2, 8), text, fill=WHITE, font=font)
        if stat["label"]:
            cd.text(((w - cd.textlength(stat["label"], font=small)) / 2, 148), stat["label"],
                    fill=CYAN, font=small)
        _blend_paste(panel, card, (x0, y0), _rounded_mask((w, h), 22, 235), p)

    # ------------------------------------------------------------ lower third
    def _lower_third(self, base, idx, vis, lt, dur):
        source = vis.get("source")
        if not source or lt < 0.6:
            return
        p_in = ease_out((lt - 0.6) / 0.5)
        p_out = _clamp((dur - lt) / EXIT)
        alpha = min(p_in, p_out)
        name = self._sprite(("src", idx), source, _font(34, True), WHITE)
        tag = self._sprite(("srct", idx), "SOURCE", _font(22, True), (8, 12, 32))
        w = tag.w + 36 + name.w + 52
        layer = Image.new("RGB", (w, 66), (10, 16, 40))
        d = ImageDraw.Draw(layer)
        d.rectangle([(0, 0), (8, 66)], fill=CYAN)
        d.rounded_rectangle([(26, 16), (26 + tag.w + 20, 50)], radius=17, fill=PURPLE)
        _blend_paste(base, layer, (int(MARGIN - (1 - p_in) * 220), 706), _rounded_mask((w, 66), 10, 238), alpha)
        x = int(MARGIN - (1 - p_in) * 220)
        tag.paste(base, x + 36, 706 + 17, alpha)
        name.paste(base, x + 26 + tag.w + 40, 706 + 12, alpha)

    # -------------------------------------------------------------- subtitles
    def _layout_cue(self, cue: Cue):
        cached = self._cue_layout.get(id(cue))
        if cached:
            return cached
        draw = ImageDraw.Draw(Image.new("RGB", (4, 4)))
        for size in (46, 42, 38):
            font = _font(size, True)
            space = draw.textlength(" ", font=font)
            max_w = WIDTH - 2 * MARGIN - 80
            lines, line, x = [], [], 0.0
            for i, w in enumerate(cue.words):
                wl = draw.textlength(w.word, font=font)
                if line and x + wl > max_w:
                    lines.append(line)
                    line, x = [], 0.0
                line.append((i, w.word, x, wl))
                x += wl + space
            if line:
                lines.append(line)
            if len(lines) <= 2:
                break
        layout = (font, size, lines)
        self._cue_layout[id(cue)] = layout
        return layout

    def _subtitles(self, base, seg, t):
        cues: List[Cue] = seg.get("cues") or []
        if not cues:
            return
        first, last = cues[0].start - 0.15, cues[-1].end + 0.35
        if not (first <= t <= last):
            return
        plate_alpha = _clamp(min((t - first) / 0.2, (last - t) / 0.2))
        img, mask = self.plate
        _blend_paste(base, img, (MARGIN, SUB_TOP), mask, plate_alpha * 0.95)
        cue = cue_at(cues, t)
        if cue is None:
            return
        font, size, lines = self._layout_cue(cue)
        active = active_word_index(cue, t)
        d = ImageDraw.Draw(base)
        step = int(size * 1.34)
        block_h = len(lines) * step
        y = SUB_TOP + (SUB_BOTTOM - SUB_TOP - block_h) // 2
        for line in lines:
            width = line[-1][2] + line[-1][3]
            x0 = (WIDTH - width) / 2
            for i, word, dx, wl in line:
                x = x0 + dx
                if i == active:
                    d.rounded_rectangle([(x - 10, y - 4), (x + wl + 10, y + size + 14)], radius=12, fill=CYAN)
                    d.text((x, y), word, fill=(4, 10, 28), font=font)
                else:
                    d.text((x, y), word, fill=WHITE if i < active else SOFT, font=font)
            y += step

    # ------------------------------------------------------------- transitions
    def _wipe(self, base, lt):
        if lt > 0.55:
            return
        p = lt / 0.55
        x = int(-40 + (WIDTH + 80) * ease_out(p))
        d = ImageDraw.Draw(base)
        fade = 1 - p
        d.rectangle([(x - 18, 0), (x, HEIGHT)], fill=_scale(CYAN, 0.5 + 0.5 * fade))
        d.rectangle([(x - 36, 0), (x - 18, HEIGHT)], fill=_scale(PURPLE, 0.35 + 0.4 * fade))
        d.rectangle([(x - 90, 0), (x - 36, HEIGHT)], fill=_scale(NAVY_BOTTOM, 0.9 + fade * 0.6))

    # ---------------------------------------------------------------- timeline
    def _timeline(self, base, t):
        scenes = [s for s in self.segments if s["type"] == "scene"]
        if not scenes:
            return
        start, end = scenes[0]["start"], scenes[-1]["end"]
        span = max(end - start, 0.1)
        d = ImageDraw.Draw(base)
        x0, x1 = MARGIN, WIDTH - MARGIN
        d.rounded_rectangle([(x0, TIMELINE_Y), (x1, TIMELINE_Y + 6)], radius=3, fill=LINE)
        frac = _clamp((t - start) / span)
        xp = x0 + (x1 - x0) * frac
        if xp > x0 + 2:
            d.rounded_rectangle([(x0, TIMELINE_Y), (xp, TIMELINE_Y + 6)], radius=3, fill=CYAN)
        for seg in scenes[1:]:
            xm = x0 + (x1 - x0) * (seg["start"] - start) / span
            d.ellipse([(xm - 6, TIMELINE_Y - 3), (xm + 6, TIMELINE_Y + 9)],
                      fill=CYAN if seg["start"] <= t else LINE)
        d.ellipse([(xp - 9, TIMELINE_Y - 6), (xp + 9, TIMELINE_Y + 12)], fill=PURPLE)

    # ------------------------------------------------------------ bumper/close
    def _centered(self, base, key, text, font, color, y, alpha, rise=0):
        sprite = self._sprite(key, text, font, color)
        sprite.paste(base, (WIDTH - sprite.w) / 2, y + rise, alpha)
        return sprite

    def _bumper(self, base, lt, dur):
        fade = _clamp((dur - lt) / 0.35)
        d = ImageDraw.Draw(base)
        cx, cy = WIDTH // 2, 400
        for delay, col, rmax in ((0.0, CYAN, 300), (0.25, PURPLE, 380)):
            p = _clamp((lt - delay) / 1.1)
            if 0 < p < 1:
                r = rmax * ease_out(p)
                d.ellipse([(cx - r, cy - r), (cx + r, cy + r)], outline=_scale(col, (1 - p) * fade),
                          width=max(1, int(7 * (1 - p))))
        mark = ease_out((lt - 0.2) / 0.6) * fade
        if mark > 0:
            r = 62 * mark
            d.ellipse([(cx - r, cy - r), (cx + r, cy + r)], outline=_scale(CYAN, fade), width=9)
            r2 = 24 * mark
            d.ellipse([(cx - r2, cy - r2), (cx + r2, cy + r2)], fill=_scale(PURPLE, fade))
        a1 = ease_out((lt - 0.5) / 0.6) * fade
        self._centered(base, "b1", "AI DAILY", _font(150, True), WHITE, 500, a1, (1 - a1) * 40)
        a2 = ease_out((lt - 0.8) / 0.6) * fade
        self._centered(base, "b2", "INTELLIGENCE", _font(104, True), CYAN, 660, a2, (1 - a2) * 40)
        bar = ease_out((lt - 1.0) / 0.7)
        if bar > 0:
            w = int(760 * bar)
            d.rectangle([(cx - w // 2, 800), (cx + w // 2, 806)], fill=_scale(PURPLE, fade))
        a3 = ease_out((lt - 1.3) / 0.6) * fade
        self._centered(base, "b3", f"YOUR DAILY AI BRIEFING  |  {self.date}", _font(36), SOFT, 840, a3)

    def _closing(self, base, lt, dur):
        fade = _clamp((dur - lt) / 0.6)
        a0 = ease_out(lt / 0.6) * fade
        self._centered(base, "c0", "THAT'S YOUR AI BRIEFING", _font(88, True), WHITE, 170, a0, (1 - a0) * 40)
        d = ImageDraw.Draw(base)
        bar = int(620 * ease_out((lt - 0.3) / 0.7))
        if bar > 0:
            d.rectangle([(WIDTH // 2 - bar // 2, 290), (WIDTH // 2 + bar // 2, 296)], fill=_scale(CYAN, fade))
        headlines = [clean_display_text(s.get("headline", "")) for s in self.plan.get("stories", [])
                     if isinstance(s, dict) and s.get("headline")][:4]
        y = 350
        for i, text in enumerate(headlines):
            a = ease_out((lt - 0.6 - i * 0.18) / 0.5) * fade
            short = text if len(text) <= 78 else text[:75].rsplit(" ", 1)[0] + "..."
            num = self._sprite(("cn", i), f"{i + 1:02d}", _font(34, True), PURPLE)
            line = self._sprite(("cl", i, short), short, _font(36), SOFT)
            x = (WIDTH - (num.w + 28 + line.w)) / 2
            num.paste(base, x, y + (1 - a) * 20, a)
            line.paste(base, x + num.w + 28, y + (1 - a) * 20, a)
            y += 78
        a = ease_out((lt - 1.4) / 0.6) * fade
        self._centered(base, "c1", "See you tomorrow  |  Full report in today's PDF", _font(38, True),
                       CYAN, 780, a)
        if self.voice_credit:
            self._centered(base, ("cv", self.voice_credit), self.voice_credit, _font(26), MUTED, 960, a)