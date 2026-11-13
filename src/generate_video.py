
import argparse
import json
from pathlib import Path

from src.video.config import VideoConfig
from src.video.render import RenderError
from src.video.renderer import DEFAULT_VIDEO_DIR, generate_daily_video
from src.video.audio_mix import MixError
from src.video.tts import TTSError


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PROGRAMME_FILE = PROJECT_ROOT / "output" / "daily_programme.json"


def _load_json(path: Path):
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _attach_collection_data(programme: dict, programme_path: Path) -> None:
    """Attach existing collection data for three-minute source coverage."""
    output_dir = programme_path.resolve().parent

    updates_path = output_dir / "updates.json"
    report_path = output_dir / "collection_report.json"

    updates = _load_json(updates_path)
    report = _load_json(report_path)

    if isinstance(updates, list):
        articles = updates
    elif isinstance(updates, dict):
        articles = next(
            (
                updates[key]
                for key in ("articles", "updates", "items", "stories")
                if isinstance(updates.get(key), list)
            ),
            [],
        )
    else:
        articles = []

    if isinstance(report, dict):
        # Some collection reports carry article metadata missing from updates.json.
        known_urls = {
            item.get("url") for item in articles
            if isinstance(item, dict) and isinstance(item.get("url"), str)
        }
        for source_record in report.get("sources", []):
            if not isinstance(source_record, dict):
                continue
            source_name = source_record.get("source") or source_record.get("name")
            for article in source_record.get("articles", []):
                if isinstance(article, dict):
                    item = dict(article)
                    item.setdefault("source", source_name)
                    url = item.get("url")
                    if isinstance(url, str) and url in known_urls:
                        continue
                    articles.append(item)
                    if isinstance(url, str):
                        known_urls.add(url)

    programme["source_articles"] = [
        article for article in articles if isinstance(article, dict)
    ]
    programme["collection_report"] = report if isinstance(report, dict) else {}

    print(
        f"Three-minute source data loaded: "
        f"{len(programme['source_articles'])} articles"
    )


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Generate daily AI news video and synchronized assets"
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_PROGRAMME_FILE)
    parser.add_argument(
        "--type",
        choices=["one_minute", "three_minute"],
        default="one_minute",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_VIDEO_DIR)

    parser.add_argument(
        "--tts",
        choices=["auto", "elevenlabs", "windows"],
        default=None,
    )
    parser.add_argument("--voice-id", default=None)
    parser.add_argument(
        "--music",
        choices=["generated", "file", "off"],
        default=None,
    )
    parser.add_argument("--music-file", default=None)
    parser.add_argument("--image-dir", default=None)

    args = parser.parse_args(argv)

    if not args.input.exists():
        parser.error(f"Input programme file not found: {args.input}")

    with args.input.open("r", encoding="utf-8") as file:
        programme = json.load(file)

    # Only the three-minute video receives additional collection data.
    # The one-minute workflow remains unchanged.
    if args.type == "three_minute":
        _attach_collection_data(programme, args.input)

    config = VideoConfig.from_env(
        tts_provider=args.tts,
        elevenlabs_voice_id=args.voice_id,
        music_mode=args.music,
        music_file=args.music_file,
        image_dir=args.image_dir,
    )

    if args.music_file and args.music is None:
        config = VideoConfig.from_env(
            tts_provider=args.tts,
            elevenlabs_voice_id=args.voice_id,
            music_mode="file",
            music_file=args.music_file,
            image_dir=args.image_dir,
        )

    try:
        result = generate_daily_video(
            programme,
            programme_type=args.type,
            output_dir=args.output_dir,
            config=config,
        )
    except TTSError as exc:
        print(f"\nVOICE ERROR [{exc.kind}]: {exc}")
        print("No video was written. Fix the voice problem and run again.")
        print("Test Windows speech with: python -m src.video.voiceover")
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
    print(f"Preview stills: {len(result['frames'])}")

    if result["video_path"]:
        print(f"Compiled MP4 Video: {result['video_path']}")
    else:
        print("MP4 was not rendered.")

    return result


if __name__ == "__main__":
    main()
