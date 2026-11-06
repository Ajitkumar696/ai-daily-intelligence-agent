import argparse
import json
from pathlib import Path

from src.video.config import VideoConfig
from src.video.render import RenderError
from src.video.renderer import DEFAULT_VIDEO_DIR, generate_daily_video
from src.video.audio_mix import MixError
from src.video.tts import TTSError


DEFAULT_PROGRAMME_FILE = (
    Path(__file__).resolve().parent.parent / "output" / "daily_programme.json"
)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Generate daily AI news video and synchronized assets from daily_programme.json"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_PROGRAMME_FILE,
        help="Path to daily_programme.json"
    )
    parser.add_argument(
        "--type",
        choices=["one_minute", "three_minute"],
        default="one_minute",
        help="Programme type to render (default: one_minute)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_VIDEO_DIR,
        help="Output directory for video assets"
    )

    parser.add_argument("--tts", choices=["auto", "elevenlabs", "windows"], default=None,
                        help="Voice: auto (ElevenLabs if a key is set, else Windows), "
                             "elevenlabs, or windows (default: VIDEO_TTS_PROVIDER or auto)")
    parser.add_argument("--voice-id", default=None, help="ElevenLabs voice id (ELEVENLABS_VOICE_ID)")
    parser.add_argument("--music", choices=["generated", "file", "off"], default=None,
                        help="Background music: generated, file (see --music-file) or off")
    parser.add_argument("--music-file", default=None,
                        help="Your own royalty-free/licensed music file (use with --music file)")
    parser.add_argument("--image-dir", default=None,
                        help="Folder with story images named <story_id>.jpg/png (optional)")

    args = parser.parse_args(argv)

    if not args.input.exists():
        parser.error(f"Input programme file not found: {args.input}")

    with args.input.open("r", encoding="utf-8") as f:
        programme = json.load(f)

    config = VideoConfig.from_env(
        tts_provider=args.tts, elevenlabs_voice_id=args.voice_id, music_mode=args.music,
        music_file=args.music_file, image_dir=args.image_dir,
    )
    if args.music_file and args.music is None:
        config = VideoConfig.from_env(
            tts_provider=args.tts, elevenlabs_voice_id=args.voice_id, music_mode="file",
            music_file=args.music_file, image_dir=args.image_dir)

    try:
        result = generate_daily_video(
            programme,
            programme_type=args.type,
            output_dir=args.output_dir,
            config=config,
        )
    except TTSError as exc:
        print(f"\nVOICE ERROR [{exc.kind}]: {exc}")
        print("No video was written. Nothing was replaced. Fix the voice problem and run again.")
        print("Test Windows speech with:  python -m src.video.voiceover")
        raise SystemExit(1)
    except (RenderError, MixError, ValueError) as exc:
        print(f"\nVIDEO ERROR: {exc}")
        raise SystemExit(1)

    print("\n" + "=" * 50)
    print("AI DAILY INTELLIGENCE VIDEO GENERATOR")
    print("=" * 50)
    print(f"Status: {result['status']}")
    print(f"Scenes Generated: {result['scene_count']}")
    print(f"Duration: {result['duration_seconds']}s")
    print(f"Video Spec: {result['spec_path']}")
    print(f"Voiceover Audio: {result['voiceover_path']}")
    print(f"Voice: {result.get('voice_engine')}   Music: {result.get('music')}")
    for warning in result.get("warnings", []):
        print(f"WARNING: {warning}")
    print(f"Subtitles (SRT): {result['subtitles_path']}")
    print(f"Preview stills: {len(result['frames'])} in {result['spec_path'].parent / 'frames'}")

    if result["video_path"]:
        print(f"Compiled MP4 Video: {result['video_path']}")
    else:
        print("MP4 was not rendered (render disabled).")

    return result


if __name__ == "__main__":
    main()