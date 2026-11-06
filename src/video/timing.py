"""Word timings and subtitle cues built from real (or estimated) speech timing."""
import re
from dataclasses import dataclass
from typing import List, Optional, Sequence


@dataclass(frozen=True)
class WordTiming:
    word: str
    start: float
    end: float


@dataclass
class Cue:
    words: List[WordTiming]

    @property
    def start(self) -> float:
        return self.words[0].start

    @property
    def end(self) -> float:
        return self.words[-1].end

    @property
    def text(self) -> str:
        return " ".join(w.word for w in self.words)


def words_from_alignment(characters: Sequence[str], starts: Sequence[float],
                         ends: Sequence[float]) -> List[WordTiming]:
    """Group ElevenLabs per-character timing into word timings."""
    words: List[WordTiming] = []
    current, w_start, w_end = "", 0.0, 0.0
    for ch, s, e in zip(characters, starts, ends):
        if str(ch).isspace():
            if current:
                words.append(WordTiming(current, float(w_start), float(w_end)))
                current = ""
            continue
        if not current:
            w_start = s
        current += str(ch)
        w_end = e
    if current:
        words.append(WordTiming(current, float(w_start), float(w_end)))
    return words


def estimate_word_timings(text: str, duration: float, lead: float = 0.0) -> List[WordTiming]:
    """Spread words over ``duration`` by length and punctuation.

    Used only when the speech engine gives no timing (Windows speech); the
    total is the *measured* audio length, so drift stays within a word or two.
    """
    tokens = text.split()
    if not tokens:
        return []
    weights = []
    for tok in tokens:
        w = len(tok) + 2.0
        if tok.endswith((".", "!", "?")):
            w += 5
        elif tok.endswith((",", ";", ":")):
            w += 2.5
        weights.append(w)
    total = sum(weights)
    usable = max(duration - lead, 0.1)
    out, cursor = [], lead
    for tok, w in zip(tokens, weights):
        span = usable * w / total
        out.append(WordTiming(tok, round(cursor, 3), round(cursor + span, 3)))
        cursor += span
    return out


def shift_words(words: Sequence[WordTiming], offset: float) -> List[WordTiming]:
    return [WordTiming(w.word, round(w.start + offset, 3), round(w.end + offset, 3))
            for w in words]


def align_to_text(words: Sequence[WordTiming], text: str, duration: float,
                  lead: float = 0.0) -> List[WordTiming]:
    """Make sure the timed words are exactly the narration's words.

    Provider alignment can differ slightly (e.g. normalised numbers); the
    subtitles must show the narration verbatim, so on any mismatch we fall
    back to the estimate rather than showing different text.
    """
    expected = text.split()
    if [w.word for w in words] == expected:
        return list(words)
    return estimate_word_timings(text, duration, lead)


def build_cues(words: Sequence[WordTiming], max_words: int = 9,
               max_chars: int = 52) -> List[Cue]:
    """Split a scene's words into short, readable subtitle phrases."""
    cues: List[Cue] = []
    current: List[WordTiming] = []
    chars = 0
    for word in words:
        current.append(word)
        chars += len(word.word) + 1
        ends_phrase = word.word.endswith((".", "!", "?", ";", ":"))
        soft_break = word.word.endswith(",") and (len(current) >= 4 or chars >= max_chars * 0.6)
        if ends_phrase or soft_break or len(current) >= max_words or chars >= max_chars:
            cues.append(Cue(current))
            current, chars = [], 0
    if current:
        if cues and len(current) <= 2 and len(cues[-1].words) + len(current) <= max_words + 2:
            cues[-1].words.extend(current)  # avoid a one-word orphan cue
        else:
            cues.append(Cue(current))
    return cues


def active_word_index(cue: Cue, t: float) -> int:
    """Index of the word being spoken at time ``t`` (last started word)."""
    index = -1
    for i, word in enumerate(cue.words):
        if t >= word.start:
            index = i
        else:
            break
    return index


def cue_at(cues: Sequence[Cue], t: float, hold: float = 0.25) -> Optional[Cue]:
    """The cue to show at ``t`` (kept briefly after its last word, never long)."""
    for i, cue in enumerate(cues):
        nxt = cues[i + 1].start - 0.05 if i + 1 < len(cues) else cue.end + hold
        limit = min(nxt, cue.end + 1.0)
        if cue.start - 0.05 <= t < limit:
            return cue
    return None


def _ts(seconds: float) -> str:
    ms = int(round(seconds * 1000))
    return f"{ms // 3600000:02d}:{ms % 3600000 // 60000:02d}:{ms % 60000 // 1000:02d},{ms % 1000:03d}"


def cues_to_srt(cues: Sequence[Cue]) -> str:
    blocks = []
    for i, cue in enumerate(cues, 1):
        end = cue.end + 0.15
        if i < len(cues):
            end = min(end, cues[i].start)
        blocks.append(f"{i}\n{_ts(cue.start)} --> {_ts(max(end, cue.start + 0.2))}\n{cue.text}\n")
    return "\n".join(blocks).strip()