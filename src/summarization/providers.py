import copy
import json
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import os
import re
import time
from pathlib import Path
from urllib.parse import urlsplit

import requests
from dotenv import load_dotenv

from .gemini_client import (
    GeminiClientError,
    GeminiConfigurationError,
)
from .token_budget import (
    GroqTokenBudget,
    TokenBudgetExceeded,
    estimate_request_tokens,
)


GROQ_DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_DEFAULT_MODEL = "openai/gpt-oss-20b"
GROQ_DEFAULT_MAX_OUTPUT_TOKENS = 4096
GROQ_TIMEOUT_SECONDS = 90
GROQ_RETRY_DELAYS_SECONDS = (5.0, 15.0, 45.0, 90.0)
GROQ_TRANSIENT_STATUS_CODES = {429, 500, 502, 503, 504}
# Longest server-suggested wait (rate-limit hint) worth sleeping through.
GROQ_MAX_SERVER_RETRY_WAIT_SECONDS = 120
# Groq's documented models with strict Structured Outputs support.
GROQ_STRICT_JSON_SCHEMA_MODELS = frozenset({
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
    "qwen/qwen3.8-27b",
    "meta-models/muse-spark-open",
})


class GroqProviderError(GeminiClientError):
    """Sanitized request or response error from Groq."""

    def __init__(
        self,
        message,
        status_code=None,
        error_type=None,
        transient=False,
        stage=None,
        api_detail=None,
    ):
        super().__init__(message, status_code, error_type, transient)
        self.stage = stage
        self.api_detail = api_detail


class GroqConfigurationError(GeminiConfigurationError):
    """Invalid local Groq API settings."""


_RETRY_IN_PATTERN = re.compile(
    r"try again in\s+(?:(\d+(?:\.\d+)?)h)?\s*(?:(\d+(?:\.\d+)?)m)?\s*"
    r"(?:(\d+(?:\.\d+)?)s)?",
    re.IGNORECASE,
)
_REPORTED_USED_PATTERN = re.compile(r"\bUsed\s*[:=]?\s*([\d,]+)", re.IGNORECASE)


def _is_daily_token_quota(response):
    detail = _safe_api_error_detail(response, "")
    return bool(re.search(r"\b(?:TPD|tokens?\s+per\s+day|daily\s+(?:token\s+)?limit)\b", detail, re.IGNORECASE))


def _reported_used_tokens(response):
    detail = _safe_api_error_detail(response, "")
    match = _REPORTED_USED_PATTERN.search(detail)
    return int(match.group(1).replace(",", "")) if match else None


def _server_retry_hint_seconds(response):
    """Seconds Groq asked us to wait, or None when it gave no usable hint.

    Reads the Retry-After header, then the "Please try again in 44.085s"
    sentence in the error message. Never raises.
    """
    try:
        headers = getattr(response, "headers", None) or {}
        value = (
            headers.get("retry-after")
            or headers.get("Retry-After")
            or headers.get("x-ratelimit-reset-tokens")
        )
        if value is not None:
            try:
                return max(0.0, float(value))
            except (TypeError, ValueError):
                try:
                    reset_at = parsedate_to_datetime(str(value))
                    return max(
                        0.0,
                        (reset_at - datetime.now(timezone.utc)).total_seconds(),
                    )
                except (TypeError, ValueError, OverflowError):
                    duration = re.fullmatch(
                        r"(?:(\d+(?:\.\d+)?)h)?\s*(?:(\d+(?:\.\d+)?)m)?\s*"
                        r"(?:(\d+(?:\.\d+)?)s)?",
                        str(value).strip(), re.IGNORECASE,
                    )
                    if duration and any(duration.groups()):
                        hours, minutes, seconds = (
                            float(part or 0) for part in duration.groups()
                        )
                        return hours * 3600 + minutes * 60 + seconds
    except (TypeError, ValueError, AttributeError):
        pass
    try:
        body = response.json()
        message = (body.get("error") or {}).get("message", "")
        match = _RETRY_IN_PATTERN.search(str(message))
    except (ValueError, TypeError, AttributeError):
        return None
    if not match or not any(match.groups()):
        return None
    hours, minutes, seconds = (float(part or 0) for part in match.groups())
    return hours * 3600 + minutes * 60 + seconds


def _safe_api_error_detail(response, api_key):
    """Return bounded, redacted provider diagnostics, never the raw body."""
    try:
        body = response.json()
    except (ValueError, TypeError, AttributeError):
        return ""
    error = body.get("error") if isinstance(body, dict) else None
    if not isinstance(error, dict):
        return ""
    parts = []
    for key in ("type", "code", "message", "failed_generation"):
        value = error.get(key)
        if isinstance(value, (str, int)) and str(value).strip():
            value = str(value).strip()
            limit = 320 if key in ("message", "failed_generation") else 100
            parts.append(f"{key}={value[:limit]}")
    detail = "; ".join(parts)
    for secret in (api_key, os.getenv("GROQ_API_KEY", "").strip()):
        if secret:
            detail = detail.replace(secret, "[REDACTED]")
    detail = re.sub(
        r"(?i)\bBearer\s+\S+", "Bearer [REDACTED]", detail
    )
    detail = re.sub(
        r"(?i)\b(gsk_[A-Za-z0-9_-]{8,})\b", "[REDACTED]", detail
    )
    detail = re.sub(
        r"(?i)\b(?:sk|xai|AIza)[_-][A-Za-z0-9_-]{12,}\b",
        "[REDACTED]",
        detail,
    )
    detail = re.sub(
        r"(?im)^\s*(?:authorization|(?:api[_ -]?)?key)\s*[:=].*$",
        "[REDACTED CREDENTIAL FIELD]",
        detail,
    )
    return " ".join(detail.split())[:900]


def load_groq_settings():
    load_dotenv(
        dotenv_path=Path(__file__).resolve().parents[2] / ".env",
        override=False,
    )
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    base_url = os.getenv("GROQ_BASE_URL", GROQ_DEFAULT_BASE_URL).strip().rstrip("/")
    model = os.getenv("GROQ_MODEL", GROQ_DEFAULT_MODEL).strip()

    if not api_key or any(character.isspace() for character in api_key) or any(
        marker in api_key.casefold() for marker in ("your_", "placeholder")
    ):
        raise GroqConfigurationError(
            "GROQ_API_KEY is missing or is an example value. Set your Groq API key."
        )
    parsed = urlsplit(base_url)
    if parsed.scheme != "https" or not parsed.netloc or parsed.query or parsed.fragment:
        raise GroqConfigurationError(
            "GROQ_BASE_URL must be an HTTPS OpenAI-compatible API base URL."
        )
    if not model or any(character.isspace() for character in model):
        raise GroqConfigurationError(
            "GROQ_MODEL must be a model ID without whitespace; default is "
            f"{GROQ_DEFAULT_MODEL}."
        )
    return api_key, base_url, model


def _strict_schema(schema):
    result = copy.deepcopy(schema)
    if not isinstance(result, dict):
        raise GroqConfigurationError("A JSON object schema is required.")

    def update(node):
        if not isinstance(node, dict):
            return
        if node.get("type") == "object":
            properties = node.get("properties", {})
            node["required"] = list(properties)
            node["additionalProperties"] = False
            for value in properties.values():
                update(value)
        if node.get("type") == "array":
            update(node.get("items"))
        for key in ("anyOf", "oneOf", "allOf"):
            for item in node.get(key, []):
                update(item)

    update(result)
    return result


class GroqClient:
    provider = "groq"

    def __init__(
        self,
        api_key,
        model=GROQ_DEFAULT_MODEL,
        max_output_tokens=GROQ_DEFAULT_MAX_OUTPUT_TOKENS,
        base_url=GROQ_DEFAULT_BASE_URL,
        max_retries=4,
        retry_delays=None,
        session=None,
        token_budget=None,
    ):
        if not api_key or not model or not base_url:
            raise GroqConfigurationError(
                "GROQ_API_KEY, GROQ_BASE_URL, and GROQ_MODEL are required."
            )
        self._api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.max_output_tokens = max_output_tokens
        if not isinstance(max_retries, int) or not 0 <= max_retries <= len(
            GROQ_RETRY_DELAYS_SECONDS
        ):
            raise GroqConfigurationError("Groq retries must be between 0 and 4.")
        self.max_retries = max_retries
        self.retry_delays = tuple(retry_delays or GROQ_RETRY_DELAYS_SECONDS)
        self._session = session or requests
        self.token_budget = token_budget or GroqTokenBudget.from_environment()
        self.last_usage = {}
        self.api_call_count = 0
        self.last_stage = None

    def generate_daily_programme(self, prompt):
        from ..research.daily_programme import DAILY_PROGRAMME_SCHEMA
        return self.generate_json(
            prompt, DAILY_PROGRAMME_SCHEMA, strict=False, stage="daily_programme"
        )

    def generate_daily_story_batch(self, prompt):
        if self.model in GROQ_STRICT_JSON_SCHEMA_MODELS:
            from ..research.daily_programme import DAILY_STORY_BATCH_SCHEMA
            return self.generate_json(
                prompt,
                DAILY_STORY_BATCH_SCHEMA,
                strict=True,
                stage="story_batch",
            )
        return self.generate_json_object(prompt, stage="story_batch")

    def generate_programme_scripts(self, prompt, schema):
        return self.generate_json(
            prompt, schema, strict=True, stage="final_script"
        )

    def generate_narration(self, prompt):
        from ..research.daily_programme import NARRATION_SCHEMA
        return self.generate_json(
            prompt, NARRATION_SCHEMA, stage="narration"
        )

    def generate_research_output(self, prompt):
        from ..research.schemas import RESEARCH_OUTPUT_JSON_SCHEMA
        return self.generate_json(
            prompt, RESEARCH_OUTPUT_JSON_SCHEMA, stage="research_output"
        )

    def generate_summary(self, title, source_content):
        from .schemas import SUMMARY_JSON_SCHEMA, validate_summary_data

        prompt = (
            "Summarize this single source item using only the supplied "
            "content. Do not add facts, dates, statistics, capabilities, "
            "or announcements that are not explicitly present. Do not "
            "return a title; Python restores the collected article title. "
            "If a detail is unclear, say "
            "'Not specified in the source.' in that field. Keep the "
            "summary concise and key points directly supported by the text.\n\n"
            f"Source title: {title}\n\nSource content:\n{source_content}"
        )
        try:
            return validate_summary_data(
                self.generate_json(
                    prompt, SUMMARY_JSON_SCHEMA, stage="summary"
                ),
                trusted_title=title,
            )
        except ValueError as error:
            raise GroqProviderError(
                f"Groq returned invalid summary data: {error}",
                error_type="INVALID_SCHEMA",
                stage="summary",
            ) from error

    def generate_json(self, prompt, schema, strict=True, stage="structured_json"):
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_completion_tokens": self.max_output_tokens,
            "temperature": 0,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "daily_intelligence_response",
                    "strict": bool(strict),
                    "schema": _strict_schema(schema),
                },
            },
        }
        return self._request_json(payload, stage)

    def generate_json_object(self, prompt, stage):
        payload = {
            "model": self.model,
            "messages": [{
                "role": "user",
                "content": (
                    "Return valid JSON only, with no Markdown fences or "
                    "surrounding prose.\n\n" + prompt
                ),
            }],
            "max_completion_tokens": self.max_output_tokens,
            "temperature": 0,
            "response_format": {"type": "json_object"},
        }
        return self._request_json(payload, stage)

    def _request_json(self, payload, stage):
        self.last_stage = stage
        # gpt-oss models spend output tokens on hidden reasoning. Keep it low so
        # the JSON is not truncated at max_completion_tokens (which produced
        # json_validate_failed errors and wasted tokens). If Groq rejects this
        # parameter with HTTP 400, delete these two lines.
        if self.model.startswith("openai/gpt-oss"):
            payload.setdefault("reasoning_effort", "low")
        url = self.base_url + "/chat/completions"
        estimate = estimate_request_tokens(payload)
        for attempt in range(self.max_retries + 1):
            try:
                self.token_budget.assert_can_spend(self.model, stage, estimate)
            except TokenBudgetExceeded as error:
                raise GroqProviderError(
                    str(error), error_type="TOKEN_BUDGET_EXHAUSTED", stage=stage
                ) from error
            try:
                self.api_call_count += 1
                response = self._session.post(
                    url,
                    headers={
                        "Authorization": f"Bearer {self._api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=GROQ_TIMEOUT_SECONDS,
                )
            except requests.RequestException as error:
                self.token_budget.record_estimated_usage(self.model, estimate)
                if attempt < self.max_retries:
                    time.sleep(self.retry_delays[min(attempt, len(self.retry_delays) - 1)])
                    continue
                raise GroqProviderError(
                    f"Groq {stage} request failed ({type(error).__name__}); check network "
                    "access and GROQ_BASE_URL.",
                    error_type=type(error).__name__,
                    transient=True,
                    stage=stage,
                ) from error

            status = response.status_code
            hint = _server_retry_hint_seconds(response)
            if status == 429 and _is_daily_token_quota(response):
                self.token_budget.reconcile_quota(
                    self.model,
                    reported_used=_reported_used_tokens(response),
                    reset_after=hint,
                )
                detail = _safe_api_error_detail(response, self._api_key)
                raise GroqProviderError(
                    f"Groq daily token quota exhausted at {stage}; no further "
                    "requests will be sent until its reset."
                    + (f" API detail: {detail}" if detail else ""),
                    status_code=429,
                    error_type="DAILY_QUOTA_EXHAUSTED",
                    transient=False,
                    stage=stage,
                    api_detail=detail,
                )
            if status in (500, 502, 503, 504):
                self.token_budget.record_estimated_usage(self.model, estimate)
            if status in GROQ_TRANSIENT_STATUS_CODES and attempt < self.max_retries:
                delay = self.retry_delays[min(attempt, len(self.retry_delays) - 1)]
                # A hint longer than the cap (e.g. a daily limit) is not worth
                # waiting for: fall through and report the error now.
                if hint is None or hint <= GROQ_MAX_SERVER_RETRY_WAIT_SECONDS:
                    if hint is not None:
                        delay = max(delay, hint + 1)
                    time.sleep(delay)
                    continue
            if not response.ok:
                hints = {
                    401: " Check that GROQ_API_KEY is valid.",
                    403: " Check model access and account permissions.",
                    404: " Check GROQ_BASE_URL and GROQ_MODEL.",
                    429: " Reduce request volume or retry after the rate limit resets.",
                }
                hint = hints.get(
                    status,
                    " Check Groq service status and retry later."
                    if status in GROQ_TRANSIENT_STATUS_CODES
                    else "",
                )
                detail = _safe_api_error_detail(response, self._api_key)
                raise GroqProviderError(
                    f"Groq {stage} API error HTTP {status}.{hint}"
                    + (f" API detail: {detail}" if detail else ""),
                    status_code=status,
                    error_type="HTTP_ERROR",
                    transient=status in GROQ_TRANSIENT_STATUS_CODES,
                    stage=stage,
                    api_detail=detail,
                )

            try:
                result = response.json()
                choice = result["choices"][0]
                content = choice["message"]["content"]
            except (ValueError, KeyError, IndexError, TypeError) as error:
                raise GroqProviderError(
                    f"Groq {stage} returned an invalid API response or no choice content.",
                    error_type="INVALID_API_RESPONSE",
                    stage=stage,
                ) from error
            if not isinstance(content, str) or not content.strip():
                raise GroqProviderError(
                    f"Groq {stage} returned empty response content.",
                    error_type="EMPTY_RESPONSE",
                    stage=stage,
                )
            self.last_usage = {
                key: value
                for key, value in (result.get("usage") or {}).items()
                if key in ("prompt_tokens", "completion_tokens", "total_tokens")
                and isinstance(value, int)
            }
            self.last_usage["finish_reason"] = choice.get("finish_reason")
            self.token_budget.record_usage(self.model, self.last_usage)
            try:
                parsed = json.loads(content)
            except json.JSONDecodeError as error:
                raise GroqProviderError(
                    f"Groq {stage} returned invalid JSON.",
                    error_type="INVALID_JSON",
                    stage=stage,
                ) from error
            if stage == "story_batch":
                parsed = _normalize_story_batch_json(parsed)
            if not isinstance(parsed, dict):
                top_level_type = (
                    "null" if parsed is None
                    else "boolean" if isinstance(parsed, bool)
                    else "number" if isinstance(parsed, (int, float))
                    else "array" if isinstance(parsed, list)
                    else "string" if isinstance(parsed, str)
                    else type(parsed).__name__
                )
                raise GroqProviderError(
                    f"Groq {stage} returned JSON with top-level type="
                    f"{top_level_type}; expected object.",
                    error_type="INVALID_API_RESPONSE",
                    stage=stage,
                )
            return parsed


def _normalize_story_batch_json(parsed):
    """Wrap a bare array of research objects as {"stories": [...]}.

    Only a non-empty array whose items are all objects is accepted. Anything
    else (empty array, scalars, null) is returned unchanged so the normal
    top-level check reports it. Positional mapping, the exact-count check and
    the non-content-field check all happen downstream in the batch validator,
    so this does not loosen what is accepted as research content.
    """
    if (
        isinstance(parsed, list)
        and parsed
        and all(isinstance(item, dict) for item in parsed)
    ):
        if len(parsed) == 1 and isinstance(parsed[0].get("stories"), list):
            return parsed[0]
        return {"stories": parsed}
    return parsed


def create_llm_client():
    api_key, base_url, model = load_groq_settings()
    return GroqClient(api_key, model, base_url=base_url)


def check_groq_configuration():
    """Check Groq settings without making a request."""
    _, base_url, model = load_groq_settings()
    return {
        "ready": True,
        "provider": "groq",
        "model": model,
        "base_url": base_url,
    }