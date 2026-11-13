import json
from pathlib import Path
import pytest
from PIL import Image

from src.video.planner import plan_video_scenes, _format_srt_timestamp
from src.video.voiceover import synthesize_voiceover
from src.video.renderer import generate_daily_video, render_scene_frame


class FakeVoice:
    """Test voice: writes a modulated tone (never used outside tests)."""
    name = "Test voice"

    def __init__(self, seconds_per_word=0.4):
        self.seconds_per_word = seconds_per_word

    def synthesize(self, text, out, lead, tail, previous_text="", next_text=""):
        import math, struct, wave
        from src.video import tts
        from src.video.timing import estimate_word_timings, shift_words
        speech = len(text.split()) * self.seconds_per_word
        rate = tts.SAMPLE_RATE
        n = int(round((lead + speech + tail) * rate))
        buf = bytearray()
        for i in range(n):
            t = i / rate
            v = int(8000 * math.sin(2 * math.pi * 180 * t)) if lead <= t < lead + speech else 0
            buf += struct.pack("<h", v)
        with wave.open(str(out), "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(rate); w.writeframes(bytes(buf))
        words = shift_words(estimate_word_timings(text, speech), lead)
        return tts.SceneAudio(out, n / rate, words, self.name, True)


def fake_providers():
    return {"windows": FakeVoice()}


def _test_config(**kw):
    from src.video.config import VideoConfig
    return VideoConfig.from_env({}, tts_provider="windows", music_mode="off", fps=10, **kw)


@pytest.fixture
def sample_programme():
    return {
        "programme_date": "2026-11-04",
        "overall_headline": "AI Researchers Introduce New Generative Breakthrough",
        "stories": [
            {
                "story_id": "story-001",
                "topic": "Neural Vector Synthesis",
                "headline": "Breakthrough in Vector Graphics with Agentic Models",
                "why_it_matters": "Enables scalable resolution-independent graphics from text.",
                "key_facts": [
                    {"statement": "Model uses iterative agentic refinement."}
                ],
                "source_urls": ["https://example.com/paper/001"]
            }
        ],
        "programmes": {
            "one_minute": {
                "title": "Daily AI News: Breakthroughs in Agentic Generation",
                "duration_seconds": 60,
                "narration": (
                    "Welcome to your Daily AI Update. I'm your host. "
                    "Today, researchers introduced a new framework for vector sketch generation. "
                    "Unlike traditional tools, it mimics human sketching using an iterative loop. "
                    "The team also released a benchmark dataset with thousands of editable assets. "
                    "That wraps up your quick AI briefing for today. Thanks for tuning in."
                ),
                "word_count": 60,
                "story_ids": ["story-001"],
                "source_urls": ["https://example.com/paper/001"]
            },
            "three_minute": {
                "title": "Comprehensive AI News Bulletin",
                "duration_seconds": 180,
                "narration": (
                    "Welcome to the Daily AI News Programme. Today we bring you detailed coverage. "
                    "Generative models have traditionally struggled with editable vector drawings. "
                    "A new agentic framework breaks prompts into Bézier curves and refines them. "
                    "The framework demonstrates state of the art results across all benchmarks. "
                    "Researchers have also made the entire dataset publicly available for study. "
                    "This opens up new horizons for graphic design workflows and automation. "
                    "Thank you for listening, and join us again tomorrow for more insights."
                ),
                "word_count": 80,
                "story_ids": ["story-001"],
                "source_urls": ["https://example.com/paper/001"]
            }
        }
    }


def test_plan_video_scenes_preserves_exact_narration(sample_programme):
    for p_type in ("one_minute", "three_minute"):
        plan = plan_video_scenes(sample_programme, programme_type=p_type)
        expected_narration = sample_programme["programmes"][p_type]["narration"].strip()

        # 1. Exact narration preservation
        assert plan["narration"] == expected_narration
        reconstructed = " ".join(scene["narration"] for scene in plan["scenes"]).strip()
        assert reconstructed == expected_narration

        # 2. Timing consistency
        assert plan["actual_duration_seconds"] == sample_programme["programmes"][p_type]["duration_seconds"]
        assert len(plan["scenes"]) >= 3

        # 3. Contiguous time spans
        prev_end = 0.0
        for scene in plan["scenes"]:
            assert scene["start_seconds"] == pytest.approx(prev_end)
            assert scene["end_seconds"] > scene["start_seconds"]
            prev_end = scene["end_seconds"]

        # 4. Subtitles match
        assert len(plan["subtitles"]) == len(plan["scenes"])
        assert "-->" in plan["subtitles_srt"]


def test_plan_video_scenes_rejects_invalid_inputs(sample_programme):
    with pytest.raises(ValueError, match="must be a dictionary"):
        plan_video_scenes(None)

    with pytest.raises(ValueError, match="not found in daily programme"):
        plan_video_scenes(sample_programme, programme_type="five_minute")

    invalid = {"programmes": {"one_minute": {"narration": ""}}}
    with pytest.raises(ValueError, match="No narration found"):
        plan_video_scenes(invalid, programme_type="one_minute")


def test_format_srt_timestamp():
    assert _format_srt_timestamp(0.0) == "00:00:00,000"
    assert _format_srt_timestamp(65.25) == "00:01:05,250"
    assert _format_srt_timestamp(3661.5) == "01:01:01,500"


def test_voiceover_synthesis_creates_valid_wav_file(tmp_path):
    output_wav = tmp_path / "voiceover.wav"
    result = synthesize_voiceover("Testing speech synthesis output.", output_wav)
    assert result.exists()
    assert result.stat().st_size > 100


def test_render_scene_frame_creates_1080p_image(sample_programme, tmp_path):
    plan = plan_video_scenes(sample_programme, "one_minute")
    frame_path = tmp_path / "frame_01.png"
    rendered = render_scene_frame(plan["scenes"][0], plan, frame_path)

    assert rendered.exists()
    with Image.open(rendered) as img:
        assert img.size == (1920, 1080)
        assert img.mode == "RGB"


def test_generate_daily_video_end_to_end(sample_programme, tmp_path):
    result = generate_daily_video(
        sample_programme,
        programme_type="one_minute",
        output_dir=tmp_path / "videos",
        render_mp4_if_possible=False,
        config=_test_config(),
        providers=fake_providers(),
        log=lambda *_: None,
    )

    assert result["status"] in ("SUCCESS_MP4_CREATED", "SUCCESS_VIDEO_PACKAGE_READY")
    assert result["spec_path"].exists()
    assert result["subtitles_path"].exists()
    assert result["voiceover_path"].exists()
    assert len(result["frames"]) == result["scene_count"]
    for frame in result["frames"]:
        assert frame.exists()

    with result["spec_path"].open("r", encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded["programme_type"] == "one_minute"
    assert loaded["actual_duration_seconds"] == pytest.approx(
        result["duration_seconds"], abs=0.01)


# --- readable frames, story matching, audio-synced timing ------------------

def _multi_story_programme():
    return {
        "programme_date": "2026-11-05",
        "stories": [
            {
                "story_id": "s1", "topic": "AI Safety",
                "headline": "Anthropic updates independent evaluators",
                "summary": "Independent evaluators assessed models fairly.",
                "key_facts": [{"statement": "Independent evaluators help ensure models are assessed fairly."}],
                "why_it_matters": "Fairer assessment.",
                "source_urls": ["https://www.anthropic.com/news/evaluators"],
                "sources": [{"publisher": "Anthropic", "url": "https://www.anthropic.com/news/evaluators"}],
            },
            {
                "story_id": "s2", "topic": "Efficiency",
                "headline": "DeepMem cuts language model compute",
                "summary": "DeepMem architecture reduces language model compute dramatically.",
                "key_facts": [{"statement": "DeepMem architecture reduces compute while keeping fidelity."}],
                "why_it_matters": "Cheaper deployment.",
                "source_urls": ["https://huggingface.co/papers/1"],
                "sources": [{"publisher": "Hugging Face Papers", "url": "https://huggingface.co/papers/1"}],
            },
        ],
        "programmes": {"one_minute": {
            "title": "Daily AI News",
            "duration_seconds": 60,
            "narration": (
                "Welcome to your Daily AI Update. I'm your host. "
                "First, Anthropic published an update on independent evaluators. "
                "Independent evaluators help ensure models are assessed fairly. "
                "Meanwhile, DeepMem architecture reduces language model compute dramatically. "
                "DeepMem keeps fidelity while cutting compute for deployment. "
                "That wraps up your briefing. Thanks for tuning in."
            ),
            "story_ids": ["s1", "s2"],
            "source_urls": [],
        }},
    }


def test_scenes_follow_the_story_being_narrated():
    plan = plan_video_scenes(_multi_story_programme(), "one_minute")
    story_scenes = [s for s in plan["scenes"] if s["kind"] == "story"]

    assert plan["scenes"][0]["kind"] == "intro"
    assert plan["scenes"][-1]["kind"] == "outro"
    first, second = story_scenes[0], story_scenes[-1]
    assert "Anthropic" in first["headline"] and "Anthropic" in first["narration"]
    assert "DeepMem" in second["headline"] and "DeepMem" in second["narration"]
    assert first["story_id"] == "s1" and "Anthropic" in first["source_credit"]
    assert second["story_id"] == "s2" and "Hugging Face Papers" in second["source_credit"]
    # a scene never mixes sentences about two different stories
    for scene in story_scenes:
        assert not ("Anthropic" in scene["narration"] and "DeepMem" in scene["narration"])


def test_three_minute_scenes_keep_story_publisher_and_domain_together():
    programme = _multi_story_programme()
    programme["programmes"]["three_minute"] = {
        "title": "Full bulletin", "duration_seconds": 180,
        "narration": (
            "Today, Anthropic published an update on independent evaluators. "
            "Independent evaluators help ensure models are assessed fairly. "
            "DeepMem architecture reduces language model compute dramatically. "
            "DeepMem keeps fidelity while cutting compute for deployment. "
            "That wraps up your briefing."
        ),
        "story_ids": ["s1", "s2"], "source_urls": [],
    }
    plan = plan_video_scenes(programme, "three_minute")
    scenes = [scene for scene in plan["scenes"] if scene["kind"] == "story"]

    assert scenes[0]["story_id"] == "s1"
    assert "Anthropic" in scenes[0]["source_credit"]
    assert "anthropic.com" in scenes[0]["source_credit"]
    assert scenes[-1]["story_id"] == "s2"
    assert "Hugging Face Papers" in scenes[-1]["source_credit"]
    assert "huggingface.co" in scenes[-1]["source_credit"]


def test_three_minute_video_keeps_pdf_programme_stories_when_collection_data_is_attached(sample_programme):
    programme = sample_programme
    programme["source_articles"] = [{
        "source": "Unrelated Publisher", "title": "Unrelated model launch",
        "url": "https://unrelated.example/story",
        "summary": "An unrelated product announcement with collected evidence.",
    }]
    plan = plan_video_scenes(programme, "three_minute")

    stories = [scene for scene in plan["scenes"] if scene["kind"] == "story"]
    assert stories and {scene["story_id"] for scene in stories} == {"story-001"}
    assert all("unrelated" not in scene["narration"].lower() for scene in stories)
    assert stories[0]["source_urls"] == ["https://example.com/paper/001"]


def test_three_minute_duration_guard_rejects_out_of_range_audio_but_leaves_one_minute_alone():
    from src.video.pipeline import validate_target_duration
    from src.video.render import RenderError

    validate_target_duration({"programme_type": "three_minute", "target_duration_seconds": 180}, 194)
    validate_target_duration({"programme_type": "one_minute", "target_duration_seconds": 60}, 220)
    with pytest.raises(RenderError, match="renders to 222.8s"):
        validate_target_duration(
            {"programme_type": "three_minute", "target_duration_seconds": 180}, 222.8
        )


def test_three_minute_audio_fit_is_local_and_rescales_subtitle_word_times(tmp_path, monkeypatch):
    from src.video import pipeline, tts
    from src.video.timing import WordTiming

    original = tts.write_silence(tmp_path / "scene.wav", 190)
    clip = tts.SceneAudio(original, 190, [WordTiming("Briefing.", 1, 2)], "Windows speech", True)

    def fake_ffmpeg(command, _label):
        source = Path(command[command.index("-i") + 1])
        output = Path(command[-1])
        tempo = float(command[command.index("-filter:a") + 1].split("=")[1])
        tts.write_silence(output, tts.wav_seconds(source) / tempo)

    monkeypatch.setattr(pipeline.render, "run_ffmpeg", fake_ffmpeg)
    fitted = pipeline.fit_three_minute_audio(
        [clip], _test_config(), tmp_path / "work", "ffmpeg", target_seconds=180,
    )

    assert fitted[0].path != original
    assert fitted[0].duration == pytest.approx(174, abs=0.05)
    assert fitted[0].words[0].start == pytest.approx(1 / (190 / 174))
    assert fitted[0].words[0].end == pytest.approx(2 / (190 / 174))


def test_clean_display_text_replaces_glyphs_fonts_cannot_draw():
    from src.video.planner import clean_display_text

    assert clean_display_text("byte‑level fidelity") == "byte-level fidelity"
    assert clean_display_text("  a   b \n c ") == "a b c"


def test_frames_use_a_real_font_at_the_requested_size():
    from src.video.renderer import _font

    box = _font(40).getbbox("Ag")
    assert box[3] - box[1] >= 25          # the default bitmap font is ~11 px


def test_frame_has_no_debug_labels_in_the_rendered_source():
    import inspect
    import src.video.renderer as renderer

    source = inspect.getsource(renderer.render_scene_frame)
    for label in ("VISUAL SUGGESTION", "TIMING:", "NARRATION / SUBTITLE"):
        assert label not in source


def test_scene_timing_follows_real_speech_length(tmp_path):
    from src.video.voiceover import wav_duration_seconds

    result = generate_daily_video(
        _multi_story_programme(), "one_minute", tmp_path / "videos",
        render_mp4_if_possible=False, config=_test_config(),
        providers=fake_providers(), log=lambda *_: None,
    )

    assert result["audio_synced"] is True
    spec = json.loads(result["spec_path"].read_text(encoding="utf-8"))
    audio_seconds = wav_duration_seconds(result["voiceover_path"])
    assert spec["actual_duration_seconds"] == pytest.approx(audio_seconds, abs=0.05)
    assert spec["actual_duration_seconds"] != 60
    previous_end = spec["timeline"][0]["end"]          # opening title comes first
    for scene in spec["scenes"]:
        assert scene["start_seconds"] == pytest.approx(previous_end, abs=0.01)
        previous_end = scene["end_seconds"]
    assert "-->" in result["subtitles_path"].read_text(encoding="utf-8")


def test_without_any_speech_engine_the_real_error_is_raised_not_a_fake_voice(tmp_path):
    from src.video.tts import TTSError

    class Broken:
        name = "Windows speech"

        def synthesize(self, *a, **k):
            raise TTSError("engine missing", "unavailable", "Windows speech")

    with pytest.raises(TTSError) as info:
        generate_daily_video(
            _multi_story_programme(), "one_minute", tmp_path / "videos",
            render_mp4_if_possible=False, config=_test_config(),
            providers={"windows": Broken()}, log=lambda *_: None,
        )
    assert "engine missing" in str(info.value)
    assert not list((tmp_path / "videos").rglob("voiceover.wav"))


# --- speech engines: silent output must not count as a voice ---------------

def _write_wav(path, seconds=2.0, amplitude=0):
    import struct
    import wave

    rate = 16000
    sample = struct.pack("<h", amplitude)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(sample * int(rate * seconds))
    return path


def test_wav_has_audible_speech_rejects_silence(tmp_path):
    from src.video.voiceover import wav_has_audible_speech

    assert wav_has_audible_speech(_write_wav(tmp_path / "loud.wav", amplitude=8000))
    assert not wav_has_audible_speech(_write_wav(tmp_path / "silent.wav", amplitude=0))
    assert not wav_has_audible_speech(tmp_path / "missing.wav")


def test_silent_engine_is_skipped_and_next_engine_is_used(tmp_path, monkeypatch):
    import src.video.voiceover as vo

    calls = []

    def silent_engine(text, path, rate=0):
        calls.append("silent")
        _write_wav(path, amplitude=0)
        return True

    def working_engine(text, path, rate=0):
        calls.append("working")
        _write_wav(path, amplitude=9000)
        return True

    monkeypatch.setattr(vo, "_speech_engines",
                        lambda: [("Silent", silent_engine), ("Working", working_engine)])
    out = tmp_path / "speech.wav"

    assert vo.try_native_speech("Hello there", out) is True
    assert calls == ["silent", "working"]
    assert vo.LAST_SPEECH_ENGINE == "Working"
    assert vo.wav_has_audible_speech(out)


def test_all_silent_engines_report_failure(tmp_path, monkeypatch):
    import src.video.voiceover as vo

    def silent_engine(text, path, rate=0):
        _write_wav(path, amplitude=0)
        return True

    monkeypatch.setattr(vo, "_speech_engines", lambda: [("Silent", silent_engine)])
    out = tmp_path / "speech.wav"

    assert vo.try_native_speech("Hello there", out) is False
    assert not out.exists()          # the silent file is removed, not left behind


def test_engine_exception_does_not_crash(tmp_path, monkeypatch):
    import src.video.voiceover as vo

    def broken(text, path, rate=0):
        raise RuntimeError("no COM")

    monkeypatch.setattr(vo, "_speech_engines", lambda: [("Broken", broken)])
    assert vo.try_native_speech("Hello", tmp_path / "x.wav") is False


# --- narration without a greeting, and quotes that broke PowerShell --------

def test_narration_that_starts_with_a_story_has_no_title_card_intro():
    programme = _multi_story_programme()
    programme["programmes"]["one_minute"]["narration"] = (
        "First, Anthropic published an update on independent evaluators. "
        "Independent evaluators help ensure models are assessed fairly. "
        "Meanwhile, DeepMem architecture reduces language model compute dramatically. "
        "DeepMem keeps fidelity while cutting compute for deployment. "
        "These breakthroughs show where the field is heading today."
    )
    plan = plan_video_scenes(programme, "one_minute")

    assert plan["scenes"][0]["kind"] == "story"
    assert "Anthropic" in plan["scenes"][0]["headline"]
    assert plan["scenes"][-1]["kind"] == "outro"
    reconstructed = " ".join(s["narration"] for s in plan["scenes"])
    assert reconstructed == programme["programmes"]["one_minute"]["narration"]


def test_speech_text_travels_in_a_file_not_in_the_powershell_script(tmp_path, monkeypatch):
    import pathlib
    import src.video.voiceover as vo

    text = "OpenAI’s “quoted” ‘x’ text, it's 7,600× faster."
    seen = {}

    def fake_run(cmd, **kwargs):
        script = pathlib.Path(cmd[-1])
        seen["script"] = script.read_text(encoding="utf-8-sig")
        seen["text"] = script.with_suffix(".txt").read_text(encoding="utf-8")

        class Result:
            returncode = 0
        return Result()

    monkeypatch.setattr(vo.subprocess, "run", fake_run)
    for engine in (vo._synthesize_system_speech, vo._synthesize_windows_sapi):
        assert engine(text, tmp_path / "out.wav", 0) is True
        assert seen["text"] == text                    # arrives byte-for-byte
        assert "OpenAI" not in seen["script"]          # never embedded in the script
        assert "’" not in seen["script"]


def test_powershell_quote_doubles_every_kind_of_apostrophe():
    from src.video.voiceover import _ps_quote

    assert _ps_quote("it's") == "'it''s'"
    assert _ps_quote("it’s") == "'it''s'"
    quoted = _ps_quote("‘a‚‛")
    assert quoted == "'" + "''" + "a" + "''" + "''" + "'"
    assert not any(ch in quoted for ch in "‘’‚‛")


def _source_programme(source_count=12):
    sources = [f"Publisher {index}" for index in range(source_count)]
    records = [
        {"source": name, "status": "SUCCESS", "article_count": 1}
        for name in sources
    ]
    records.extend([
        {"source": "X / Twitter", "status": "FAILED", "article_count": 0},
        {"source": "Product Hunt", "status": "FAILED", "article_count": 0},
    ])
    articles = [
        {
            "source": name,
            "publisher": name,
            "title": f"Publisher {index} announces a new AI research model",
            "url": f"https://example.com/story/{index}",
            "summary": (
                f"Publisher {index} describes a new research model for developers. "
                "The collected report says the release includes downloadable weights "
                "and benchmark results for practical evaluation."
            ),
            "content_source_type": "full_article",
        }
        for index, name in enumerate(sources)
    ]
    return {
        "programme_date": "2026-11-11",
        "stories": [],
        "source_articles": articles,
        "collection_report": {"sources": records},
        "programmes": {
            "one_minute": {
                "title": "One minute",
                "duration_seconds": 60,
                "narration": "Welcome. A source article follows. Thanks for listening.",
            },
            "three_minute": {
                "title": "Daily AI briefing",
                "duration_seconds": 180,
                "narration": "Legacy narration stays available.",
            },
        },
    }


def test_three_minute_source_editorial_coverage_and_failed_disclosure():
    plan = plan_video_scenes(_source_programme(), "three_minute")
    stories = [scene for scene in plan["scenes"] if scene["kind"] == "story"]

    assert len(stories) == 12
    assert {scene["source_credit"] for scene in stories} == {
        f"Publisher {index}" for index in range(12)
    }
    assert all(scene["source_urls"] for scene in stories)
    assert {
        scene["source_credit"]: scene["source_urls"][0]
        for scene in stories
    } == {
        f"Publisher {index}": f"https://example.com/story/{index}"
        for index in range(12)
    }
    assert all(scene["headline"] in scene["narration"] for scene in stories)
    assert all(scene["source_credit"] in scene["narration"] for scene in stories)
    assert any("X / Twitter and Product Hunt were unavailable" in scene["narration"]
               for scene in plan["scenes"])
    assert "X / Twitter" not in [scene["source_credit"] for scene in stories]
    assert 360 <= plan["word_count"] <= 420
    assert plan["actual_duration_seconds"] == 180


def test_three_minute_deduplicates_articles_and_discloses_sources_without_content():
    programme = _source_programme(source_count=2)
    duplicate = dict(programme["source_articles"][0], url="https://example.com/duplicate")
    programme["source_articles"].append(duplicate)
    programme["collection_report"]["sources"].append(
        {"source": "Quiet Source", "status": "SUCCESS", "article_count": 1}
    )
    plan = plan_video_scenes(programme, "three_minute")
    stories = [scene for scene in plan["scenes"] if scene["kind"] == "story"]

    assert len(stories) == 2
    assert len({scene["headline"] for scene in stories}) == 2
    assert any("Quiet Source" in scene["narration"] for scene in plan["scenes"])
    assert plan["source_coverage"]["sources_without_usable_coverage"] == ["Quiet Source"]


def test_three_minute_handles_nested_article_content_and_rejects_headline_only():
    programme = _source_programme(source_count=2)
    programme["source_articles"][0]["summary"] = [
        {"content": "Collected nested evidence describes a model release."},
        "The article explains the benchmark and intended use.",
    ]
    programme["source_articles"][1]["summary"] = None
    programme["source_articles"][1]["extracted_content"] = {"body": [
        {"text": "Extracted evidence describes a research result."}
    ]}
    programme["source_articles"].append({
        "source": "Publisher 0", "title": "Headline without collected evidence",
        "url": "https://example.com/headline-only",
    })
    plan = plan_video_scenes(programme, "three_minute")
    stories = [scene for scene in plan["scenes"] if scene["kind"] == "story"]

    assert len(stories) == 2
    assert "Collected nested evidence" in stories[0]["narration"]
    assert "Extracted evidence" in stories[1]["narration"]
    assert all("Headline without collected evidence" not in scene["narration"]
               for scene in stories)


def test_editorial_white_visual_is_three_minute_only():
    from src.video.motion import MotionRenderer

    programme = _source_programme(source_count=1)
    three_plan = plan_video_scenes(programme, "three_minute")
    one_plan = plan_video_scenes(programme, "one_minute")
    segment = [{"type": "scene", "index": 0, "start": 0.0, "end": 12.0, "cues": []}]
    three = MotionRenderer(three_plan, [{} for _ in three_plan["scenes"]], segment, 12)
    one = MotionRenderer(one_plan, [{} for _ in one_plan["scenes"]], segment, 12)

    assert three.render_frame(1).getpixel((0, 0)) == (250, 250, 248)
    assert max(one.render_frame(1).getpixel((0, 0))) < 60


def test_editorial_visual_concepts_follow_story_evidence():
    from src.video.motion import editorial_visual_concept

    assert editorial_visual_concept(
        {"headline": "Benchmark evaluates model safety"}, {"keywords": ["Benchmark"]}
    ) == "evaluation"
    assert editorial_visual_concept(
        {"headline": "New robot agent automates workflow"}, {"keywords": ["robot"]}
    ) == "workflow"
    assert editorial_visual_concept(
        {"headline": "Chip processor hardware update"}, {"keywords": ["processor"]}
    ) == "hardware"


def test_editorial_typography_fallback_is_story_specific_not_placeholder():
    from PIL import Image, ImageDraw
    from src.video.motion import MotionRenderer

    renderer = MotionRenderer({"programme_type": "three_minute", "scenes": []}, [], [], 1)
    first = Image.new("RGB", (1920, 1080), (234, 239, 243))
    second = first.copy()
    renderer._editorial_visual(ImageDraw.Draw(first), {"headline": "Novel discovery"},
                               {"keywords": ["Orion"]}, 1.0, (40, 105, 170),
                               (31, 38, 46), (102, 112, 122))
    renderer._editorial_visual(ImageDraw.Draw(second), {"headline": "New reasoning"},
                               {"keywords": ["Vector"]}, 1.0, (40, 105, 170),
                               (31, 38, 46), (102, 112, 122))
    assert first.crop((1012, 187, 1818, 676)).tobytes() != second.crop((1012, 187, 1818, 676)).tobytes()


def test_three_minute_pipeline_falls_back_locally_without_higgsfield_key(tmp_path):
    programme = _source_programme()
    result = generate_daily_video(
        programme,
        programme_type="three_minute",
        output_dir=tmp_path / "videos",
        render_mp4_if_possible=False,
        config=_test_config(),
        providers=fake_providers(),
        log=lambda *_: None,
    )

    spec = json.loads(result["spec_path"].read_text(encoding="utf-8"))
    assert result["status"] == "SUCCESS_VIDEO_PACKAGE_READY"
    assert 150 <= result["duration_seconds"] <= 190
    assert all("clip_path" not in visual for visual in spec["visuals"])
    assert len(result["frames"]) == result["scene_count"]


def test_higgsfield_prompt_uses_article_evidence_and_excludes_fake_claims():
    from src.video.higgsfield import build_story_prompt

    prompt = build_story_prompt({
        "headline": "Research team publishes model evaluation",
        "key_points": ["Collected article reports a new benchmark."],
    })

    assert "model evaluation" in prompt
    assert "new benchmark" in prompt
    assert "no text, logos" in prompt
    assert "news footage" not in prompt.lower()


def test_higgsfield_mocked_submission_result_and_cache(tmp_path):
    from src.video.higgsfield import MODEL_ID, generate_story_clip

    config = _test_config(
        higgsfield_enabled=True,
        higgsfield_allow_paid=True,
        higgsfield_api_key_id="test-id",
        higgsfield_api_key_secret="test-secret",
        higgsfield_clip_seconds=5,
        higgsfield_timeout=10,
    )
    calls = []

    class Client:
        def subscribe(self, model, arguments):
            calls.append((model, arguments))
            return {"video": {"url": "https://cdn.example.com/generated.mp4"}}

    class Response:
        def raise_for_status(self):
            pass

        def iter_content(self, chunk_size):
            assert chunk_size > 0
            yield b"mocked-video-bytes"

    def downloader(url, **kwargs):
        assert url == "https://cdn.example.com/generated.mp4"
        return Response()

    scene = {"headline": "New model benchmark", "key_points": ["Collected evidence explains results."]}
    clip, warning = generate_story_clip(
        scene, config, tmp_path,
        client_factory=lambda **kwargs: Client(), downloader=downloader,
    )
    cached, cached_warning = generate_story_clip(
        scene, config, tmp_path,
        client_factory=lambda **kwargs: pytest.fail("Cached clip must not be regenerated."),
        downloader=downloader,
    )

    assert warning == cached_warning == ""
    assert clip == cached and clip.read_bytes() == b"mocked-video-bytes"
    assert len(calls) == 1 and calls[0][0] == MODEL_ID
    assert calls[0][1]["sound"] == "off"
    assert calls[0][1]["duration"] == 5


def test_higgsfield_missing_credentials_and_unapproved_calls_fall_back(tmp_path):
    from src.video.higgsfield import generate_story_clip, generate_story_clips

    scene = {"headline": "Evidence based article", "key_points": ["Collected source summary."]}
    no_key = _test_config(higgsfield_enabled=True, higgsfield_allow_paid=True)
    clip, warning = generate_story_clip(
        scene, no_key, tmp_path,
        client_factory=lambda **kwargs: pytest.fail("No key means no API call."),
    )
    assert clip is None and "credentials" in warning

    config = _test_config(higgsfield_enabled=True, higgsfield_allow_paid=False)
    plan = {"programme_type": "three_minute", "scenes": [{"kind": "story", **scene}]}
    clips, warnings = generate_story_clips(
        plan, config, tmp_path,
        client_factory=lambda **kwargs: pytest.fail("Unapproved generation must not call the API."),
    )
    assert clips == [None]
    assert warnings and "not approved" in warnings[0]


def test_higgsfield_api_failure_falls_back_without_exposing_credentials(tmp_path):
    from src.video.higgsfield import generate_story_clip

    config = _test_config(
        higgsfield_enabled=True,
        higgsfield_allow_paid=True,
        higgsfield_api_key_id="sensitive-id",
        higgsfield_api_key_secret="sensitive-secret",
    )

    class BrokenClient:
        def subscribe(self, *args, **kwargs):
            raise RuntimeError("request failed with sensitive-secret")

    clip, warning = generate_story_clip(
        {"headline": "A reported model launch", "key_points": ["Collected article evidence."]},
        config, tmp_path, client_factory=lambda **kwargs: BrokenClient(),
    )

    assert clip is None
    assert "RuntimeError" in warning
    assert "sensitive-secret" not in warning


def test_asia_kolkata_zoneinfo_is_available():
    from zoneinfo import ZoneInfo

    assert ZoneInfo("Asia/Kolkata").key == "Asia/Kolkata"
