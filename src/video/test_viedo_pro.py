import base64
import copy
import hashlib
import json
import math
import shutil
import struct
import wave
from pathlib import Path

import pytest
import requests

from src.video import audio_mix, render, tts
from src.video.config import VideoConfig
from src.video.motion import MotionRenderer
from src.video.planner import clean_display_text, plan_video_scenes
from src.video.renderer import generate_daily_video
from src.video.timing import (WordTiming, build_cues, cue_at, cues_to_srt,
                              estimate_word_timings, words_from_alignment, align_to_text,
                              active_word_index)
from src.video.visuals import (build_scene_visuals, extract_keywords, extract_stats,
                               resolve_story_image, source_name)

HAS_FFMPEG = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))
needs_ffmpeg = pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg/ffprobe not installed")

SECRET = "sk_test_SUPER_SECRET_KEY_123"


class FakeVoice:
    name = "Test voice"

    def synthesize(self, text, out, lead, tail, previous_text="", next_text=""):
        speech = len(text.split()) * 0.4
        rate = tts.SAMPLE_RATE
        n = int(round((lead + speech + tail) * rate))
        buf = bytearray()
        for i in range(n):
            t = i / rate
            buf += struct.pack("<h", int(8000 * math.sin(2 * math.pi * 180 * t))
                               if lead <= t < lead + speech else 0)
        with wave.open(str(out), "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(rate); w.writeframes(bytes(buf))
        from src.video.timing import shift_words
        return tts.SceneAudio(out, n / rate, shift_words(estimate_word_timings(text, speech), lead),
                              self.name, True)


def programme():
    stories = [
        {"story_id": "story-1", "headline": "SketchDirector Automates Scene Sketching",
         "topic": "Sketching", "category": "AI Research",
         "summary": "Quality improved by 23% on the MCSketch8K benchmark.",
         "key_facts": [{"statement": "Scores rose 23% over baselines."}],
         "source_urls": ["https://huggingface.co/papers/1"], "why_it_matters": "Better sketches."},
        {"story_id": "story-2", "headline": "Anthropic Releases Faster Coding Agents",
         "topic": "Models", "category": "Model Release",
         "summary": "Latency and cost drop for coding agents.",
         "key_facts": [{"statement": "Coding agents run faster."}],
         "source_urls": ["https://www.anthropic.com/news/x"], "why_it_matters": "Cheaper agents."},
    ]
    narration = (
        "Welcome to your Daily AI Update. "
        "Today, researchers introduced SketchDirector, which automates complex scene sketching. "
        "The team reports a 23% quality gain on the MCSketch8K benchmark. "
        "Next, Anthropic released faster coding agents with lower latency and cost. "
        "Coding agents run faster for developers. "
        "That's your briefing for today."
    )
    return {"programme_date": "2026-11-05", "stories": stories,
            "programmes": {"one_minute": {"title": "Daily AI News", "duration_seconds": 60,
                                          "narration": narration}}}


def cfg(**kw):
    base = dict(tts_provider="windows", music_mode="off", fps=10)
    base.update(kw)
    return VideoConfig.from_env({}, **base)


# ------------------------------------------------------------- configuration
def test_config_reads_environment_and_never_prints_the_key():
    c = VideoConfig.from_env({"ELEVENLABS_API_KEY": SECRET, "VIDEO_TTS_PROVIDER": "elevenlabs",
                              "VIDEO_MUSIC": "off", "ELEVENLABS_VOICE_ID": "abc"})
    assert c.has_elevenlabs_key and c.tts_provider == "elevenlabs"
    assert c.music_mode == "off" and c.elevenlabs_voice_id == "abc"
    assert c.elevenlabs_model == "eleven_multilingual_v2"
    assert SECRET not in repr(c) and SECRET not in str(c)


def test_config_defaults_and_bad_values():
    c = VideoConfig.from_env({"VIDEO_TTS_PROVIDER": "nonsense", "VIDEO_MUSIC": "loud"})
    assert c.tts_provider == "auto" and c.music_mode == "generated"
    assert not c.has_elevenlabs_key


def test_provider_order():
    assert tts.provider_order(cfg(tts_provider="auto")) == ["windows"]
    assert tts.provider_order(cfg(tts_provider="auto", elevenlabs_api_key="k")) == ["elevenlabs", "windows"]
    assert tts.provider_order(cfg(tts_provider="auto", elevenlabs_api_key="k",
                                  fallback_to_windows=False)) == ["elevenlabs"]
    assert tts.provider_order(cfg(tts_provider="elevenlabs", elevenlabs_api_key="k")) == ["elevenlabs"]


# ---------------------------------------------------------------- ElevenLabs
class Resp:
    def __init__(self, status, payload):
        self.status_code, self._p = status, payload
        self.text = json.dumps(payload)

    def json(self):
        return self._p


def mp3_bytes(tmp_path):
    out = tmp_path / "x.mp3"
    import subprocess
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i",
                    "sine=frequency=220:duration=1.2", str(out)], check=True)
    return out.read_bytes()


def eleven(tmp_path, post, **kw):
    c = cfg(tts_provider="elevenlabs", elevenlabs_api_key=SECRET, **kw)
    return tts.ElevenLabsProvider(c, shutil.which("ffmpeg") or "ffmpeg", tmp_path / "cache",
                                  http_post=post, sleep=lambda s: None)


@pytest.mark.parametrize("status,payload,kind", [
    (401, {"detail": {"status": "invalid_api_key", "message": "Invalid API key"}}, "auth"),
    (401, {"detail": {"status": "quota_exceeded", "message": "You have 0 credits left"}}, "quota"),
    (402, {"detail": {"status": "paid_plan_required", "message": "Upgrade"}}, "plan"),
    (400, {"detail": {"status": "bad", "message": "nope"}}, "bad_request"),
])
def test_elevenlabs_http_errors_are_classified_and_not_retried(tmp_path, status, payload, kind):
    calls = []

    def post(url, **kw):
        calls.append(url)
        return Resp(status, payload)

    with pytest.raises(tts.TTSError) as info:
        eleven(tmp_path, post).synthesize("Hello world.", tmp_path / "a.wav", 0.3, 0.4)
    assert info.value.kind == kind
    assert len(calls) == 1
    assert SECRET not in str(info.value)


def test_elevenlabs_rate_limit_is_retried_then_reported(tmp_path):
    calls = []

    def post(url, **kw):
        calls.append(1)
        return Resp(429, {"detail": {"status": "too_many_concurrent_requests", "message": "slow"}})

    with pytest.raises(tts.TTSError) as info:
        eleven(tmp_path, post).synthesize("Hello.", tmp_path / "a.wav", 0.3, 0.4)
    assert info.value.kind == "rate_limit" and len(calls) == 3


def test_elevenlabs_network_failure_is_reported_without_the_key(tmp_path):
    def post(url, **kw):
        raise requests.ConnectionError(f"failed for {kw['headers']['xi-api-key']}")

    with pytest.raises(tts.TTSError) as info:
        eleven(tmp_path, post).synthesize("Hello.", tmp_path / "a.wav", 0.3, 0.4)
    assert info.value.kind == "network"
    assert SECRET not in str(info.value)


def test_elevenlabs_requires_a_key():
    with pytest.raises(tts.TTSError) as info:
        tts.ElevenLabsProvider(cfg(), "ffmpeg", Path("."))
    assert info.value.kind == "auth"


@needs_ffmpeg
def test_elevenlabs_success_sends_exact_text_model_and_uses_real_timestamps(tmp_path):
    audio = base64.b64encode(mp3_bytes(tmp_path)).decode()
    text = "Hello brave world."
    chars = list(text)
    starts = [i * 0.05 for i in range(len(chars))]
    ends = [s + 0.05 for s in starts]
    seen = {}

    def post(url, headers=None, params=None, json=None, timeout=None):
        seen.update(url=url, headers=headers, params=params, body=json)
        return Resp(200, {"audio_base64": audio, "alignment": {
            "characters": chars, "character_start_times_seconds": starts,
            "character_end_times_seconds": ends}})

    p = eleven(tmp_path, post)
    clip = p.synthesize(text, tmp_path / "a.wav", 0.35, 0.45, previous_text="Before.", next_text="After.")
    assert seen["body"]["text"] == text                       # narration is not rewritten
    assert seen["body"]["model_id"] == "eleven_multilingual_v2"
    assert seen["body"]["previous_text"] == "Before." and seen["body"]["next_text"] == "After."
    assert seen["headers"]["xi-api-key"] == SECRET
    assert "/v1/text-to-speech/" in seen["url"] and seen["url"].endswith("/with-timestamps")
    assert seen["params"]["output_format"] == "mp3_44100_128"
    assert [w.word for w in clip.words] == ["Hello", "brave", "world."]
    assert clip.words[0].start == pytest.approx(0.35, abs=1e-6)  # lead silence applied
    assert not clip.estimated_timing
    assert tts.wav_seconds(clip.path) == pytest.approx(clip.duration, abs=0.01)


@needs_ffmpeg
def test_elevenlabs_cache_avoids_paying_twice(tmp_path):
    audio = base64.b64encode(mp3_bytes(tmp_path)).decode()
    calls = []

    def post(url, **kw):
        calls.append(1)
        return Resp(200, {"audio_base64": audio, "alignment": {}})

    p = eleven(tmp_path, post)
    p.synthesize("Same text here.", tmp_path / "a.wav", 0.3, 0.4)
    p.synthesize("Same text here.", tmp_path / "b.wav", 0.3, 0.4)
    assert len(calls) == 1 and p.cache_hits == 1


# ------------------------------------------------------------ Windows fallback
def test_auto_mode_falls_back_to_windows_with_a_visible_warning(tmp_path):
    class Failing:
        name = "ElevenLabs"

        def synthesize(self, *a, **k):
            raise tts.TTSError("credits used up", "quota", "ElevenLabs")

    logs = []
    c = cfg(tts_provider="auto", elevenlabs_api_key=SECRET)
    out = tts.synthesize_scenes(["Hello there world."], c, tmp_path / "a", "ffmpeg",
                                providers={"elevenlabs": Failing(), "windows": FakeVoice()},
                                log=logs.append)
    assert out["provider"] == "Test voice"
    assert any("quota" in w and "credits used up" in w for w in out["warnings"])
    assert logs and logs[0].startswith("WARNING")


def test_explicit_elevenlabs_never_silently_switches_voice(tmp_path):
    class Failing:
        def synthesize(self, *a, **k):
            raise tts.TTSError("bad key", "auth", "ElevenLabs")

    c = cfg(tts_provider="elevenlabs", elevenlabs_api_key=SECRET)
    with pytest.raises(tts.TTSError) as info:
        tts.synthesize_scenes(["Hi."], c, tmp_path / "a", "ffmpeg",
                              providers={"elevenlabs": Failing(), "windows": FakeVoice()})
    assert "bad key" in str(info.value)


def test_windows_provider_reports_missing_voice_instead_of_placeholder(tmp_path):
    p = tts.WindowsProvider("ffmpeg", speak=lambda text, path: False)
    with pytest.raises(tts.TTSError) as info:
        p.synthesize("Hello.", tmp_path / "a.wav", 0.3, 0.4)
    assert info.value.kind == "unavailable"
    assert not (tmp_path / "a.wav").exists()


@needs_ffmpeg
def test_windows_provider_converts_to_48k_mono_and_pads(tmp_path):
    def speak(text, path):
        with wave.open(str(path), "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(22050)
            w.writeframes(b"".join(struct.pack("<h", int(6000 * math.sin(i / 7))) for i in range(22050)))
        return True

    clip = tts.WindowsProvider(shutil.which("ffmpeg"), speak=speak).synthesize(
        "One two three four.", tmp_path / "a.wav", 0.35, 0.45)
    with wave.open(str(clip.path), "rb") as w:
        assert (w.getframerate(), w.getnchannels()) == (48000, 1)
    assert clip.duration == pytest.approx(0.35 + 1.0 + 0.45, abs=0.05)
    assert clip.estimated_timing and len(clip.words) == 4


# ------------------------------------------------------ exact narration + timing
def test_narration_is_preserved_through_planning_and_subtitles(tmp_path):
    prog = programme()
    original = copy.deepcopy(prog)
    narration = prog["programmes"]["one_minute"]["narration"]
    result = generate_daily_video(prog, "one_minute", tmp_path, render_mp4_if_possible=False,
                                  config=cfg(), providers={"windows": FakeVoice()}, log=lambda *_: None)
    spec = json.loads(result["spec_path"].read_text(encoding="utf-8"))
    assert " ".join(s["narration"] for s in spec["scenes"]) == narration
    shown = " ".join(c["text"] for c in spec["subtitles"])
    assert shown.split() == narration.split()                      # subtitles = narration, verbatim
    assert prog == original                                        # input never mutated


def test_alignment_words_and_fallback_when_text_differs():
    words = words_from_alignment(list("Hi you"), [0, .1, .2, .3, .4, .5], [.1, .2, .3, .4, .5, .6])
    assert [w.word for w in words] == ["Hi", "you"]
    assert words[1].start == pytest.approx(.3)
    mismatched = align_to_text(words, "Totally different words here", 2.0)
    assert [w.word for w in mismatched] == ["Totally", "different", "words", "here"]


def test_estimated_timing_covers_the_measured_duration_in_order():
    words = estimate_word_timings("One two, three four. Five", 5.0)
    assert words[0].start == 0 and words[-1].end == pytest.approx(5.0, abs=0.01)
    assert all(a.end <= b.start + 1e-6 for a, b in zip(words, words[1:]))
    assert words[1].end - words[1].start > 0


def test_cues_are_short_ordered_and_srt_is_valid():
    text = ("This is a long narration sentence that must be split into readable phrases, "
            "and it keeps going. Another sentence follows.")
    words = estimate_word_timings(text, 12.0)
    cues = build_cues(words)
    assert len(cues) >= 3
    assert all(len(c.words) <= 11 and len(c.text) <= 70 for c in cues)
    assert [w.word for c in cues for w in c.words] == text.split()
    assert cue_at(cues, cues[1].start + 0.01) is cues[1]
    assert active_word_index(cues[0], cues[0].words[2].start + 0.01) == 2
    srt = cues_to_srt(cues)
    assert srt.startswith("1\n00:00:00,000 -->") and "\n\n2\n" in srt


# ------------------------------------------------------------- scene visuals
def test_keywords_and_real_statistics_only():
    assert extract_keywords("SketchDirector Automates GPT-5 and MCSketch8K Work") == [
        "SketchDirector", "GPT-5", "MCSketch8K"]
    stats = extract_stats("Quality rose 12.5% over baseline, costing $2.4 billion. It ran in 2026.")
    assert [(s["prefix"], s["value"], s["suffix"]) for s in stats] == [("", 12.5, "%"), ("$", 2.4, "B")]
    assert extract_stats("The 2026 release has 12 authors and 3 sections.") == []   # bare numbers are not stats


def test_scene_visuals_use_only_story_data():
    plan = plan_video_scenes(programme(), "one_minute")
    vis = build_scene_visuals(plan, plan["stories"], [])
    stories = [(s, v) for s, v in zip(plan["scenes"], vis) if s["kind"] == "story"]
    assert stories
    first_scene, first_vis = stories[0]
    assert first_vis["source"] == "Hugging Face"
    assert first_vis["stats"] and first_vis["stats"][0]["value"] == 23
    assert "SketchDirector" in first_vis["keywords"]
    second_vis = stories[-1][1]
    assert second_vis["stats"] == [] and second_vis["source"] == "Anthropic"   # nothing invented
    assert all(v["image"] is None for v in vis)                               # no image -> graphic


def test_source_name_mapping():
    assert source_name(["https://www.anthropic.com/news/x"]) == "Anthropic"
    assert source_name(["https://some-blog.example.org/a"]) == "Example"
    assert source_name([]) == ""


def test_missing_or_corrupt_image_falls_back_to_animated_graphic(tmp_path):
    story = {"story_id": "story-1"}
    assert resolve_story_image(story, [tmp_path]) is None
    (tmp_path / "story-1.png").write_bytes(b"not an image")
    path = resolve_story_image(story, [tmp_path])
    assert path == tmp_path / "story-1.png"
    plan = plan_video_scenes(programme(), "one_minute")
    vis = build_scene_visuals(plan, plan["stories"], [])
    for v, s in zip(vis, plan["scenes"]):
        if s["story_id"] == "story-1":
            v["image"] = path                                  # corrupt file
    segments = [{"type": "bumper", "start": 0, "end": 1}] + [
        {"type": "scene", "index": i, "start": 1 + i * 3, "end": 4 + i * 3, "cues": []}
        for i in range(len(plan["scenes"]))]
    r = MotionRenderer(plan, vis, segments, 30)
    idx = next(i for i, s in enumerate(plan["scenes"]) if s["story_id"] == "story-1")
    frame = r.render_frame(segments[idx + 1]["start"] + 1.5)   # must not raise
    assert frame.size == (1920, 1080)


def test_real_image_gets_ken_burns_motion(tmp_path):
    from PIL import Image
    img = Image.new("RGB", (1600, 900))
    for x in range(0, 1600, 8):
        for y in range(0, 900, 60):
            img.putpixel((x, y), (x % 255, y % 255, 120))
    img.save(tmp_path / "story-1.png")
    plan = plan_video_scenes(programme(), "one_minute")
    vis = build_scene_visuals(plan, plan["stories"], [tmp_path])
    idx = next(i for i, s in enumerate(plan["scenes"]) if s["story_id"] == "story-1")
    assert vis[idx]["image"] is not None
    segments = [{"type": "bumper", "start": 0, "end": 1}] + [
        {"type": "scene", "index": i, "start": 1 + i * 4, "end": 5 + i * 4, "cues": []}
        for i in range(len(plan["scenes"]))]
    r = MotionRenderer(plan, vis, segments, 40)
    s = segments[idx + 1]["start"]
    a, b = r.render_frame(s + 1.0), r.render_frame(s + 3.0)
    assert a.tobytes() != b.tobytes()                          # the picture moves


def test_frames_animate_and_stay_1080p():
    plan = plan_video_scenes(programme(), "one_minute")
    vis = build_scene_visuals(plan, plan["stories"], [])
    segments = [{"type": "bumper", "start": 0, "end": 2.6}] + [
        {"type": "scene", "index": i, "start": 2.6 + i * 4, "end": 6.6 + i * 4, "cues": []}
        for i in range(len(plan["scenes"]))]
    end = segments[-1]["end"]
    segments.append({"type": "closing", "start": end, "end": end + 3.4})
    r = MotionRenderer(plan, vis, segments, end + 3.4)
    frames = [r.render_frame(t) for t in (0.5, 1.5, 3.0, 4.2, end + 1)]
    assert all(f.size == (1920, 1080) and f.mode == "RGB" for f in frames)
    assert len({hashlib.md5(f.tobytes()).hexdigest() for f in frames}) == len(frames)


# --------------------------------------------------- command construction
def test_video_command_uses_modern_flags():
    cmd = render.build_video_command("ffmpeg", Path("o.mp4"), 30)
    assert cmd[cmd.index("-fps_mode") + 1] == "cfr"
    assert "-vsync" not in cmd
    assert cmd[cmd.index("-c:v") + 1] == "libx264"
    assert cmd[cmd.index("-s") + 1] == "1920x1080"
    assert cmd[cmd.index("-framerate") + 1] == "30" and cmd[cmd.index("-r") + 1] == "30"
    assert "yuv420p" in cmd[cmd.index("-vf") + 1]
    assert "-an" in cmd and cmd[-1] == "o.mp4"


def test_mux_command_encodes_aac_and_copies_video():
    cmd = render.build_mux_command("ffmpeg", Path("v.mp4"), Path("a.wav"), Path("o.mp4"), 12.5)
    assert cmd[cmd.index("-c:v") + 1] == "copy" and cmd[cmd.index("-c:a") + 1] == "aac"
    assert cmd[cmd.index("-t") + 1] == "12.500" and "-vsync" not in cmd


def test_mix_command_ducks_normalises_fades_and_limits():
    args, desc, _ = audio_mix.resolve_music(cfg(music_mode="generated"), 60)
    cmd = audio_mix.build_mix_command("ffmpeg", Path("v.wav"), Path("o.wav"), 60, args, 0.16)
    graph = cmd[cmd.index("-filter_complex") + 1]
    for needle in ("loudnorm", "sidechaincompress", "afade=t=in", "afade=t=out", "alimiter", "amix"):
        assert needle in graph
    assert desc == "generated pad" and "lavfi" in cmd


def test_music_can_be_disabled_and_missing_file_falls_back():
    args, desc, warn = audio_mix.resolve_music(cfg(music_mode="off"), 30)
    assert args is None and desc == "off"
    cmd = audio_mix.build_mix_command("ffmpeg", Path("v.wav"), Path("o.wav"), 30, None)
    assert "sidechaincompress" not in " ".join(cmd) and "loudnorm" in " ".join(cmd)
    args, desc, warn = audio_mix.resolve_music(cfg(music_mode="file", music_file="/no/such.mp3"), 30)
    assert desc == "generated pad" and warn and "not found" in warn[0]
    f = Path(__file__)
    args, desc, warn = audio_mix.resolve_music(cfg(music_mode="file", music_file=str(f)), 30)
    assert desc.startswith("file:") and "-stream_loop" in args and not warn


# ----------------------------------------------------------- sync + validation
def test_voice_track_length_equals_video_timeline(tmp_path):
    plan = plan_video_scenes(programme(), "one_minute")
    c = cfg()
    audio = [FakeVoice().synthesize(s["narration"], tmp_path / f"s{i}.wav", c.scene_lead_seconds,
                                    c.scene_tail_seconds) for i, s in enumerate(plan["scenes"])]
    from src.video.pipeline import assemble_voice_track, build_segments
    segments, cues, total = build_segments(plan, audio, c)
    voice = assemble_voice_track(audio, c, tmp_path / "voice.wav", tmp_path / "w")
    assert tts.wav_seconds(voice) == pytest.approx(total, abs=0.01)
    for seg, clip in zip(segments[1:-1], audio):                  # each scene = its audio, back to back
        assert seg["end"] - seg["start"] == pytest.approx(clip.duration, abs=0.002)
    assert segments[1]["start"] == pytest.approx(c.bumper_seconds)
    assert segments[-1]["end"] == pytest.approx(total)
    assert all(segments[1]["start"] <= cue.start for cue in cues)  # no subtitle during the title card
    first_scene_words = segments[1]["cues"][0].words[0]
    assert first_scene_words.start >= segments[1]["start"] + c.scene_lead_seconds - 0.01


def good_info():
    return {"streams": [
        {"codec_type": "video", "codec_name": "h264", "width": 1920, "height": 1080,
         "avg_frame_rate": "30/1", "pix_fmt": "yuv420p"},
        {"codec_type": "audio", "codec_name": "aac"}], "format": {"duration": "60.0"}}


def test_check_probe_accepts_good_and_rejects_bad_files():
    assert render.check_probe(good_info(), 60.0, 30) == []
    bad = good_info()
    bad["streams"][0].update(codec_name="mpeg4", width=1280, height=720, avg_frame_rate="25/1")
    bad["streams"] = [bad["streams"][0]]
    bad["format"]["duration"] = "10"
    problems = " | ".join(render.check_probe(bad, 60.0, 30))
    for needle in ("h264", "1280x720", "frame rate", "no audio", "duration"):
        assert needle in problems


def test_validate_rejects_missing_and_tiny_files(tmp_path):
    assert render.validate_mp4(tmp_path / "none.mp4", 5, 30, "ffmpeg", "ffprobe")
    tiny = tmp_path / "t.mp4"
    tiny.write_bytes(b"x")
    assert render.validate_mp4(tiny, 5, 30, "ffmpeg", "ffprobe")


# ------------------------------------------------------ real end-to-end render
@needs_ffmpeg
def test_full_render_produces_valid_mp4_and_keeps_inputs(tmp_path):
    prog = programme()
    before = json.dumps(prog, sort_keys=True)
    out = tmp_path / "videos"
    result = generate_daily_video(prog, "one_minute", out, config=cfg(music_mode="generated"),
                                  providers={"windows": FakeVoice()}, log=lambda *_: None)
    assert result["status"] == "SUCCESS_MP4_CREATED"
    mp4 = result["video_path"]
    assert render.validate_mp4(mp4, result["duration_seconds"], 10, shutil.which("ffmpeg"),
                               shutil.which("ffprobe")) == []
    assert not list(out.rglob("*.partial.mp4"))                  # temp files cleaned up
    assert not (mp4.parent / "work").exists()
    assert json.dumps(prog, sort_keys=True) == before


@needs_ffmpeg
def test_failed_render_never_replaces_an_existing_valid_mp4(tmp_path, monkeypatch):
    out = tmp_path / "videos"
    prog = programme()
    first = generate_daily_video(prog, "one_minute", out, config=cfg(),
                                 providers={"windows": FakeVoice()}, log=lambda *_: None)
    mp4 = first["video_path"]
    original = mp4.read_bytes()
    marker = mp4.parent / "keep_me.txt"
    marker.write_text("existing output")

    monkeypatch.setattr(render, "validate_mp4", lambda *a, **k: ["forced failure"])
    with pytest.raises(render.RenderError) as info:
        generate_daily_video(prog, "one_minute", out, config=cfg(),
                             providers={"windows": FakeVoice()}, log=lambda *_: None)
    assert "forced failure" in str(info.value)
    assert mp4.read_bytes() == original
    assert marker.read_text() == "existing output"
    assert not list(out.rglob("*.partial.mp4"))


def test_video_pipeline_does_not_touch_the_research_files(tmp_path):
    research = tmp_path / "output"
    research.mkdir()
    files = {n: (research / n) for n in ("daily_programme.json", "processed_history.json",
                                         "summary_cache.json")}
    for n, p in files.items():
        p.write_text(json.dumps({"name": n}))
    snap = {n: hashlib.md5(p.read_bytes()).hexdigest() for n, p in files.items()}
    generate_daily_video(programme(), "one_minute", research / "videos", render_mp4_if_possible=False,
                         config=cfg(), providers={"windows": FakeVoice()}, log=lambda *_: None)
    assert snap == {n: hashlib.md5(p.read_bytes()).hexdigest() for n, p in files.items()}