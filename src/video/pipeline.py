"""Professional video pipeline, as independent stages.

 1 read/validate programme   2 scene plan        3 story visuals
 4 narration audio           5 subtitle timing   6 animated scene assets
 7 render video              8 mix audio         9 validate + atomic replace

Research output is only read, never changed. Existing files in the output
folder are never deleted; the final MP4 is only replaced by a validated one.
"""
import json
import shutil
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from . import audio_mix, render, tts
from .config import VideoConfig, default_image_dir
from .planner import apply_scene_durations, plan_video_scenes
from .timing import Cue, build_cues, cues_to_srt, shift_words
from .visuals import build_scene_visuals


def build_segments(plan: Dict[str, Any], audio: List[tts.SceneAudio], config: VideoConfig):
    """Stage 5: absolute timeline (bumper, scenes, closing) and subtitle cues."""
    apply_scene_durations(plan, [a.duration for a in audio])
    offset = config.bumper_seconds
    segments: List[Dict[str, Any]] = [{"type": "bumper", "start": 0.0, "end": offset}]
    all_cues: List[Cue] = []
    for i, (scene, clip) in enumerate(zip(plan["scenes"], audio)):
        scene["start_seconds"] = round(scene["start_seconds"] + offset, 3)
        scene["end_seconds"] = round(scene["end_seconds"] + offset, 3)
        words = shift_words(clip.words, scene["start_seconds"])
        cues = build_cues(words)
        all_cues.extend(cues)
        segments.append({"type": "scene", "index": i, "start": scene["start_seconds"],
                         "end": scene["end_seconds"], "cues": cues})
    scenes_end = segments[-1]["end"]
    total = round(scenes_end + config.closing_seconds, 3)
    segments.append({"type": "closing", "start": scenes_end, "end": total})
    plan["actual_duration_seconds"] = total
    plan["subtitles_srt"] = cues_to_srt(all_cues)
    plan["subtitles"] = [{"index": n, "start_seconds": round(c.start, 3),
                          "end_seconds": round(c.end, 3), "text": c.text}
                         for n, c in enumerate(all_cues, 1)]
    return segments, all_cues, total


def assemble_voice_track(audio: List[tts.SceneAudio], config: VideoConfig,
                         out_path: Path, work: Path) -> Path:
    """Voice timeline = silence over the opening + every scene's audio + silence over the closing."""
    work.mkdir(parents=True, exist_ok=True)
    head = tts.write_silence(work / "head.wav", config.bumper_seconds)
    tail = tts.write_silence(work / "tail.wav", config.closing_seconds)
    return tts.join_wavs([head] + [a.path for a in audio] + [tail], out_path)


def run_pipeline(
    programme: Dict[str, Any],
    programme_type: str = "one_minute",
    output_dir: Optional[Path] = None,
    render_mp4_if_possible: bool = True,
    config: Optional[VideoConfig] = None,
    providers: Optional[Dict[str, Any]] = None,
    log: Callable[[str], None] = print,
) -> Dict[str, Any]:
    from .renderer import DEFAULT_VIDEO_DIR  # late import: renderer imports this module lazily
    from .motion import MotionRenderer
    from PIL import Image

    config = config or VideoConfig.from_env()
    warnings: List[str] = []

    # 1-2. validate programme + scene plan (planner verifies the narration is preserved exactly)
    plan = plan_video_scenes(programme, programme_type=programme_type)
    date_str = plan.get("programme_date", "latest")
    base_dir = Path(output_dir) if output_dir is not None else DEFAULT_VIDEO_DIR
    target = base_dir / f"AI_Daily_Intelligence_{date_str}_{programme_type}"
    target.mkdir(parents=True, exist_ok=True)
    frames_dir = target / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    work = target / "work"
    work.mkdir(parents=True, exist_ok=True)

    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if not ffmpeg:
        raise render.RenderError("FFmpeg was not found in PATH; it is needed for audio and video.")

    # 3. story visuals (supplied images only; otherwise the animated graphic)
    search_dirs = [Path(config.image_dir)] if config.image_dir else []
    search_dirs.append(default_image_dir())
    visuals = build_scene_visuals(plan, plan.get("stories", []), search_dirs)

    # 4. narration audio: real speech or a clear error (never a placeholder tone)
    log(f"[1/5] Generating narration for {len(plan['scenes'])} scenes...")
    speech = tts.synthesize_scenes([s["narration"] for s in plan["scenes"]], config,
                                   work / "audio_scenes", ffmpeg, providers=providers, log=log)
    audio: List[tts.SceneAudio] = speech["audio"]
    provider = speech["provider"]
    warnings.extend(speech["warnings"])
    if any(a.estimated_timing for a in audio):
        warnings.append("Word highlight timing is estimated from each scene's measured audio "
                        "length (this voice provides no word timestamps).")

    # 5. timeline + subtitles
    segments, _, total = build_segments(plan, audio, config)
    srt_path = target / "subtitles.srt"
    srt_path.write_text(plan["subtitles_srt"], encoding="utf-8")
    voiceover_path = target / "voiceover.wav"
    assemble_voice_track(audio, config, voiceover_path, work)

    # 6. animated scene assets (preview stills come from the real animated renderer)
    credit = ("Voice: ElevenLabs (elevenlabs.io)" if provider == "ElevenLabs"
              else f"Voice: {provider}")
    motion = MotionRenderer(plan, visuals, segments, total, voice_credit=credit)
    frame_paths: List[Path] = []
    for i, scene in enumerate(plan["scenes"]):
        seg = segments[i + 1]
        t = seg["start"] + min(1.8, (seg["end"] - seg["start"]) * 0.6)
        path = frames_dir / f"scene_{scene['scene_number']:02d}.png"
        motion.render_frame(t).save(path, format="PNG")
        frame_paths.append(path)

    plan["timeline"] = [{k: v for k, v in s.items() if k != "cues"} for s in segments]
    plan["visuals"] = [{**v, "image": str(v["image"]) if v.get("image") else None} for v in visuals]
    plan["voice"] = {"provider": provider, "word_timing": "estimated" if any(
        a.estimated_timing for a in audio) else "measured"}
    spec_path = target / "daily_video_spec.json"
    video_path = target / f"AI_Daily_Intelligence_{date_str}.mp4"
    music_desc = config.music_mode
    mp4_ok = False
    problems: List[str] = []

    if render_mp4_if_possible:
        partial = target / f"{video_path.stem}.partial.mp4"
        silent = work / "silent_video.mp4"
        mixed = work / "mixed_audio.wav"
        try:
            # 7. render video
            log(f"[2/5] Rendering {render.frame_count(total, config.fps)} animated frames "
                f"({total:.1f}s at {config.fps} fps)...")
            render.render_silent_video(
                ffmpeg, motion.render_frame, total, silent, fps=config.fps,
                progress=lambda i, n: log(f"      frame {i}/{n}") if i % (config.fps * 10) == 0 else None)
            # 8. mix audio
            log("[3/5] Mixing narration and music...")
            music_desc, mix_warnings = audio_mix.mix_audio(ffmpeg, voiceover_path, mixed, total, config)
            warnings.extend(mix_warnings)
            log("[4/5] Muxing H.264 + AAC...")
            render.run_ffmpeg(render.build_mux_command(ffmpeg, silent, mixed, partial, total), "Mux")
            # 9. validate, then atomically replace
            log("[5/5] Validating MP4...")
            problems = render.validate_mp4(partial, total, config.fps, ffmpeg, ffprobe)
            if not problems:
                render.replace_atomically(partial, video_path)
                mp4_ok = True
        finally:
            partial.unlink(missing_ok=True)  # only our own temp files
            silent.unlink(missing_ok=True)
            mixed.unlink(missing_ok=True)
    plan["audio_mix"] = {"music": music_desc}
    plan["warnings"] = warnings
    with spec_path.open("w", encoding="utf-8") as f:
        json.dump(plan, f, indent=4, ensure_ascii=False)
    shutil.rmtree(work, ignore_errors=True)
    if problems:
        raise render.RenderError("The new MP4 failed validation and was not saved "
                                 "(any earlier MP4 is untouched): " + "; ".join(problems))

    return {
        "status": "SUCCESS_MP4_CREATED" if mp4_ok else "SUCCESS_VIDEO_PACKAGE_READY",
        "programme_date": date_str,
        "programme_type": programme_type,
        "duration_seconds": total,
        "scene_count": len(plan["scenes"]),
        "spec_path": spec_path,
        "subtitles_path": srt_path,
        "voiceover_path": voiceover_path,
        "frames": frame_paths,
        "video_path": video_path if mp4_ok else None,
        "ffmpeg_available": True,
        "audio_synced": True,
        "voice_engine": provider,
        "music": music_desc,
        "warnings": warnings,
    }