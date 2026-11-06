import hashlib
import json
from pathlib import Path

from ..processing.json_storage import atomic_write_json


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE_FILE = PROJECT_ROOT / "output" / "summary_cache.json"


def content_fingerprint(title, content):
    payload = json.dumps(
        {"title": title, "content": content},
        ensure_ascii=False,
        sort_keys=True
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class SummaryCache:
    def __init__(self, path=None):
        self.path = Path(path) if path is not None else DEFAULT_CACHE_FILE
        self.entries = self._load()

    def _load(self):
        if not self.path.exists():
            return {}

        try:
            with self.path.open("r", encoding="utf-8") as file:
                cache = json.load(file)
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(
                f"Unable to read valid summary cache '{self.path}': {error}"
            ) from error

        if (
            not isinstance(cache, dict)
            or cache.get("version") != 1
            or not isinstance(cache.get("entries"), dict)
        ):
            raise ValueError(
                f"Summary cache '{self.path}' has an invalid structure."
            )

        return cache["entries"]

    @staticmethod
    def _key(source_url, fingerprint):
        return f"{source_url}\n{fingerprint}"

    def get(self, source_url, fingerprint):
        value = self.entries.get(self._key(source_url, fingerprint))
        return dict(value) if isinstance(value, dict) else None

    def set(self, source_url, fingerprint, summary):
        self.entries[self._key(source_url, fingerprint)] = dict(summary)
        atomic_write_json(
            self.path,
            {"version": 1, "entries": self.entries}
        )
