"""Persistent local safeguards for Groq's free-tier daily token allowance."""

import json
import math
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

from ..processing.json_storage import atomic_write_json


DEFAULT_DAILY_TOKEN_QUOTA = 200_000
DEFAULT_DAILY_SAFETY_MARGIN = 40_000
DEFAULT_DAILY_TOKEN_BUDGET = (
    DEFAULT_DAILY_TOKEN_QUOTA - DEFAULT_DAILY_SAFETY_MARGIN
)
DEFAULT_FINAL_SCRIPT_RESERVE = 24_000
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_USAGE_FILE = PROJECT_ROOT / "output" / "groq_token_usage.json"


def estimate_request_tokens(payload):
    """Conservatively estimate prompt/schema tokens plus reserved completion."""
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    prompt_tokens = math.ceil(len(encoded.encode("utf-8")) / 2)
    output_tokens = payload.get("max_completion_tokens", 0)
    return prompt_tokens + (output_tokens if isinstance(output_tokens, int) else 0)


class TokenBudgetExceeded(RuntimeError):
    def __init__(self, message, *, remaining=0, estimate=0, stage=""):
        super().__init__(message)
        self.remaining = remaining
        self.estimate = estimate
        self.stage = stage


class GroqTokenBudget:
    """Tracks actual daily usage and reserves tokens for final script calls."""

    def __init__(
        self,
        path=None,
        daily_limit=DEFAULT_DAILY_TOKEN_BUDGET,
        final_script_reserve=DEFAULT_FINAL_SCRIPT_RESERVE,
        clock=None,
    ):
        self.path = Path(path) if path is not None else DEFAULT_USAGE_FILE
        self.daily_limit = int(daily_limit)
        self.final_script_reserve = int(final_script_reserve)
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        if self.daily_limit <= 0 or not 0 <= self.final_script_reserve < self.daily_limit:
            raise ValueError("Groq token budget and final-script reserve are invalid.")
        self._state = self._load()

    @classmethod
    def from_environment(cls):
        quota = int(os.getenv("GROQ_DAILY_TOKEN_QUOTA", DEFAULT_DAILY_TOKEN_QUOTA))
        safety_margin = int(os.getenv(
            "GROQ_DAILY_TOKEN_SAFETY_MARGIN", DEFAULT_DAILY_SAFETY_MARGIN
        ))
        default_budget = max(1, quota - safety_margin)
        return cls(
            path=os.getenv("GROQ_TOKEN_BUDGET_FILE") or DEFAULT_USAGE_FILE,
            daily_limit=int(os.getenv("GROQ_DAILY_TOKEN_BUDGET", default_budget)),
            final_script_reserve=int(os.getenv(
                "GROQ_FINAL_SCRIPT_TOKEN_RESERVE", DEFAULT_FINAL_SCRIPT_RESERVE
            )),
        )

    def _load(self):
        if not self.path.exists():
            return {"version": 1, "days": {}}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"Invalid Groq usage state: {error}") from error
        if not isinstance(data, dict) or data.get("version") != 1 or not isinstance(data.get("days"), dict):
            raise ValueError("Invalid Groq usage state structure.")
        return data

    def _entry(self, model):
        day = self._clock().astimezone(timezone.utc).date().isoformat()
        key = f"{day}:{model}"
        inherited_block = max((
            float(value.get("blocked_until", 0))
            for old_key, value in self._state["days"].items()
            if old_key.endswith(f":{model}") and isinstance(value, dict)
        ), default=0)
        entry = self._state["days"].setdefault(
            key, {"used": 0, "blocked_until": inherited_block}
        )
        return key, entry

    def _save(self):
        atomic_write_json(self.path, self._state)

    def used(self, model):
        return int(self._entry(model)[1].get("used", 0))

    def assert_can_spend(self, model, stage, estimate):
        _, entry = self._entry(model)
        remaining = max(0, self.daily_limit - int(entry.get("used", 0)))
        now = self._clock().timestamp()
        blocked_until = float(entry.get("blocked_until", 0))
        if blocked_until > now:
            wait = int(blocked_until - now)
            raise TokenBudgetExceeded(
                f"Groq daily token quota is blocked for {wait}s; remaining work "
                f"was not sent (stage={stage}, estimated_tokens={estimate}).",
                remaining=remaining, estimate=estimate, stage=stage,
            )
        reserve = 0 if stage == "final_script" else self.final_script_reserve
        available = max(0, remaining - reserve)
        if estimate > available:
            raise TokenBudgetExceeded(
                f"Groq token budget is insufficient for remaining work "
                f"(stage={stage}, estimated_tokens={estimate}, available={available}, "
                f"daily_remaining={remaining}, final_script_reserve={reserve}).",
                remaining=remaining, estimate=estimate, stage=stage,
            )

    def record_usage(self, model, usage):
        total = usage.get("total_tokens")
        if not isinstance(total, int) or total < 0:
            prompt = usage.get("prompt_tokens", 0)
            completion = usage.get("completion_tokens", 0)
            total = (prompt if isinstance(prompt, int) else 0) + (
                completion if isinstance(completion, int) else 0
            )
        _, entry = self._entry(model)
        entry["used"] = int(entry.get("used", 0)) + total
        self._save()

    def record_estimated_usage(self, model, estimate):
        _, entry = self._entry(model)
        entry["used"] = int(entry.get("used", 0)) + max(0, int(estimate))
        self._save()

    def reconcile_quota(self, model, reported_used=None, reset_after=None):
        _, entry = self._entry(model)
        if isinstance(reported_used, int) and reported_used >= 0:
            entry["used"] = max(int(entry.get("used", 0)), reported_used)
        now = self._clock()
        block_until = now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
        if isinstance(reset_after, (int, float)) and reset_after > 0:
            block_until = now + timedelta(seconds=reset_after)
        entry["blocked_until"] = block_until.timestamp()
        self._save()
