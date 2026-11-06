import json
import math
import os
import random
import re
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from importlib.util import find_spec
from pathlib import Path

from dotenv import load_dotenv

from ..research.schemas import RESEARCH_OUTPUT_JSON_SCHEMA
from ..research.daily_programme import (
    DAILY_PROGRAMME_SCHEMA,
    NARRATION_SCHEMA
)
from .schemas import SUMMARY_JSON_SCHEMA, validate_summary_data


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = "gemini-3.8-flash"
DEFAULT_MAX_OUTPUT_TOKENS = 8192
MAX_MODEL_OUTPUT_TOKENS = 65536
TRANSIENT_HTTP_STATUS_CODES = {429, 500, 503}
RETRY_DELAYS_SECONDS = (5.0, 15.0, 45.0, 90.0)
DEFAULT_MAX_RETRIES = len(RETRY_DELAYS_SECONDS)
DEFAULT_MAX_RETRY_WAIT_SECONDS = 300.0
MAX_GEMINI_MODEL_FALLBACKS = 3
RETRY_JITTER_RATIO = 0.2
GEMINI_25_THINKING_BUDGET = 8192
MODEL_PATTERN = re.compile(r"^gemini-[a-zA-Z0-9][a-zA-Z0-9._-]*$")
JSON_MARKDOWN_FENCE = re.compile(
    r"\A\s*```(?:json)?[ \t]*\r?\n(?P<json>.*?)\r?\n```[ \t]*\s*\Z",
    re.IGNORECASE | re.DOTALL
)


class GeminiClientError(RuntimeError):
    def __init__(
        self,
        message,
        status_code=None,
        error_type=None,
        transient=False
    ):
        super().__init__(message)
        self.status_code = status_code
        self.error_type = error_type
        self.transient = transient


class GeminiAuthenticationError(GeminiClientError):
    pass


class GeminiPermissionError(GeminiClientError):
    pass


class GeminiBillingError(GeminiClientError):
    pass


class GeminiModelError(GeminiClientError):
    pass


class GeminiRequestError(GeminiClientError):
    pass


class GeminiRateLimitError(GeminiRequestError):
    pass


class GeminiConfigurationError(GeminiClientError):
    pass


def _validate_model(model):
    if not isinstance(model, str) or not MODEL_PATTERN.fullmatch(model.strip()):
        raise GeminiConfigurationError(
            "GEMINI_MODEL must be a Gemini model ID such as "
            f"{DEFAULT_MODEL}."
        )
    return model.strip()


def _validate_api_key(api_key):
    if not api_key:
        raise GeminiAuthenticationError(
            "GEMINI_API_KEY is not configured. Add it to the project-root "
            ".env file or set it in the process environment."
        )
    if (
        any(character.isspace() for character in api_key)
        or "your_" in api_key.casefold()
        or "placeholder" in api_key.casefold()
    ):
        raise GeminiAuthenticationError(
            "GEMINI_API_KEY is empty or contains an example placeholder. "
            "Replace it with your Gemini API key."
        )


def load_gemini_settings():
    load_dotenv(dotenv_path=PROJECT_ROOT / ".env", override=False)

    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    model = _validate_model(os.getenv("GEMINI_MODEL", DEFAULT_MODEL).strip())
    token_value = os.getenv(
        "GEMINI_MAX_OUTPUT_TOKENS",
        str(DEFAULT_MAX_OUTPUT_TOKENS)
    )

    try:
        max_output_tokens = int(token_value)
    except ValueError as error:
        raise GeminiConfigurationError(
            "GEMINI_MAX_OUTPUT_TOKENS must be a positive integer."
        ) from error

    if max_output_tokens <= 0:
        raise GeminiConfigurationError(
            "GEMINI_MAX_OUTPUT_TOKENS must be a positive integer."
        )
    if api_key:
        _validate_api_key(api_key)

    return api_key, model, max_output_tokens


def load_gemini_model_fallbacks(primary_model=None):
    """Read the optional ordered fallback model list without assuming access."""
    raw = os.getenv("GEMINI_MODEL_FALLBACKS", "")
    fallbacks = []
    for value in raw.split(","):
        value = value.strip()
        if not value:
            continue
        model = _validate_model(value)
        if not re.match(r"^gemini-(?:3(?:\.|-)|2\.5(?:\.|-))", model):
            raise GeminiConfigurationError(
                f"Fallback model {model!r} has no configured thinking "
                "compatibility profile. Use a Gemini 3 or Gemini 2.5 model."
            )
        if model != primary_model and model not in fallbacks:
            fallbacks.append(model)
    if len(fallbacks) > MAX_GEMINI_MODEL_FALLBACKS:
        raise GeminiConfigurationError(
            "GEMINI_MODEL_FALLBACKS supports at most "
            f"{MAX_GEMINI_MODEL_FALLBACKS} models."
        )
    return fallbacks


def _retry_settings():
    try:
        max_retries = int(os.getenv("GEMINI_MAX_RETRIES", DEFAULT_MAX_RETRIES))
        max_wait = float(os.getenv(
            "GEMINI_MAX_RETRY_WAIT_SECONDS",
            DEFAULT_MAX_RETRY_WAIT_SECONDS
        ))
    except (TypeError, ValueError) as error:
        raise GeminiConfigurationError(
            "GEMINI_MAX_RETRIES must be 0-4 and "
            "GEMINI_MAX_RETRY_WAIT_SECONDS must be positive."
        ) from error
    if (
        not 0 <= max_retries <= len(RETRY_DELAYS_SECONDS)
        or not math.isfinite(max_wait)
        or max_wait <= 0
    ):
        raise GeminiConfigurationError(
            "GEMINI_MAX_RETRIES must be 0-4 and "
            "GEMINI_MAX_RETRY_WAIT_SECONDS must be positive."
        )
    return max_retries, max_wait


def check_gemini_configuration():
    """Check local settings and SDK availability without making a request."""

    api_key, model, max_output_tokens = load_gemini_settings()
    try:
        sdk_installed = find_spec("google.genai") is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        sdk_installed = False

    issues = []
    if not api_key:
        issues.append(
            "GEMINI_API_KEY is missing from the process environment and "
            "project-root .env file."
        )
    if not sdk_installed:
        issues.append(
            "The Google Gen AI Python SDK is not installed in this Python "
            "environment; install the pinned project dependencies."
        )

    return {
        "ready": not issues,
        "api_key_configured": bool(api_key),
        "model": model,
        "max_output_tokens": max_output_tokens,
        "google_genai_sdk_installed": sdk_installed,
        "issues": issues
    }


class GeminiClient:
    provider = "gemini"

    def __init__(
        self,
        api_key,
        model=DEFAULT_MODEL,
        max_output_tokens=DEFAULT_MAX_OUTPUT_TOKENS,
        sdk_client=None,
        fallback_models=None,
        max_retries=None,
        max_retry_wait_seconds=None
    ):
        api_key = api_key.strip() if isinstance(api_key, str) else ""
        _validate_api_key(api_key)
        model = _validate_model(model)
        if (
            isinstance(max_output_tokens, bool)
            or not isinstance(max_output_tokens, int)
            or max_output_tokens <= 0
        ):
            raise GeminiConfigurationError(
                "GEMINI_MAX_OUTPUT_TOKENS must be a positive integer."
            )

        self.model = model
        self.primary_model = model
        self.last_successful_model = ""
        self.last_model_fallback_reason = ""
        configured_fallbacks = list(
            load_gemini_model_fallbacks(model)
            if fallback_models is None
            else fallback_models
        )
        self.fallback_models = []
        self._verified_fallback_models = set()
        for fallback_model in configured_fallbacks:
            _validate_model(fallback_model)
            if fallback_model == model or fallback_model in self.fallback_models:
                raise GeminiConfigurationError(
                    "GEMINI_MODEL_FALLBACKS must not repeat a model ID."
                )
            if not re.match(r"^gemini-(?:3(?:\.|-)|2\.5(?:\.|-))", fallback_model):
                raise GeminiConfigurationError(
                    f"Fallback model {fallback_model!r} has no configured "
                    "thinking compatibility profile."
                )
            self.fallback_models.append(fallback_model)
        if len(self.fallback_models) > MAX_GEMINI_MODEL_FALLBACKS:
            raise GeminiConfigurationError(
                "GEMINI_MODEL_FALLBACKS supports at most "
                f"{MAX_GEMINI_MODEL_FALLBACKS} models."
            )
        configured_retries, configured_wait = _retry_settings()
        self.max_retries = (
            configured_retries if max_retries is None else max_retries
        )
        self.max_retry_wait_seconds = (
            configured_wait
            if max_retry_wait_seconds is None
            else max_retry_wait_seconds
        )
        if (
            isinstance(self.max_retries, bool)
            or not isinstance(self.max_retries, int)
            or not 0 <= self.max_retries <= len(RETRY_DELAYS_SECONDS)
            or not isinstance(self.max_retry_wait_seconds, (int, float))
            or not math.isfinite(self.max_retry_wait_seconds)
            or self.max_retry_wait_seconds <= 0
        ):
            raise GeminiConfigurationError("Invalid Gemini retry configuration.")
        self.max_output_tokens = max_output_tokens
        self._api_key = api_key
        self.last_usage = {}
        if sdk_client is None:
            try:
                from google import genai
                from google.genai import types
            except ImportError as error:
                raise GeminiConfigurationError(
                    "The Google Gen AI SDK is missing. Install dependencies "
                    "from requirements.txt before using Gemini summarization."
                ) from error

            sdk_client = genai.Client(
                api_key=api_key,
                http_options=types.HttpOptions(
                    retry_options=types.HttpRetryOptions(attempts=1)
                )
            )

        self.sdk_client = sdk_client

    @classmethod
    def from_environment(cls):
        api_key, model, max_output_tokens = load_gemini_settings()
        return cls(
            api_key,
            model,
            max_output_tokens,
            fallback_models=load_gemini_model_fallbacks(model)
        )

    def generate_summary(self, title, source_content):
        prompt = (
            "Summarize this single source item using only the supplied "
            "content. Do not add facts, dates, statistics, capabilities, "
            "or announcements that are not explicitly present. Do not "
            "return a title; Python restores the collected article title. "
            "If a detail is unclear, say "
            "'Not specified in the source.' in that field. Keep the "
            "summary concise and key points directly supported by the text.\n\n"
            f"Source title: {title}\n\n"
            f"Source content:\n{source_content}"
        )

        data = self._generate_json(prompt, SUMMARY_JSON_SCHEMA)
        try:
            return validate_summary_data(data, trusted_title=title)
        except ValueError as error:
            raise GeminiRequestError(
                f"Gemini returned invalid summary data: {error}"
            ) from error

    def generate_research_output(self, prompt):
        return self._generate_json(prompt, RESEARCH_OUTPUT_JSON_SCHEMA)

    def generate_daily_programme(self, prompt):
        return self._generate_json(
            prompt,
            DAILY_PROGRAMME_SCHEMA,
            retry_on_truncation=True
        )

    def generate_narration(self, prompt):
        return self._generate_json(
            prompt,
            NARRATION_SCHEMA,
            retry_on_truncation=True
        )

    def _generate_json(
        self,
        prompt,
        response_schema,
        retry_on_truncation=False
    ):
        models = [self.primary_model, *self.fallback_models]
        self._total_retry_wait = 0.0
        self.last_model_fallback_reason = ""
        failures = []
        last_error = None
        for model_index, model in enumerate(models):
            if model_index:
                try:
                    self._verify_fallback_model(model)
                except GeminiClientError as error:
                    failures.append(f"{model}: compatibility check failed: {error}")
                    if last_error is None or not last_error.transient:
                        last_error = error
                    continue

            self.model = model
            try:
                result = self._generate_json_single_model(
                    prompt,
                    response_schema,
                    retry_on_truncation
                )
            except GeminiClientError as error:
                if model_index == 0 and (
                    not error.transient or not self.fallback_models
                ):
                    raise
                failures.append(f"{model}: {error}")
                last_error = error
                if not error.transient:
                    break
                continue

            self.model = model
            self.last_successful_model = model
            self.last_model_fallback_reason = "; ".join(failures)[:1200]
            return result

        if last_error is None:
            raise GeminiModelError("No compatible Gemini model was available.")
        if not failures:
            raise last_error
        message = "Gemini model attempts failed: " + "; ".join(failures)
        raise GeminiRequestError(
            message,
            status_code=last_error.status_code,
            error_type=last_error.error_type,
            transient=last_error.transient
        ) from last_error

    def _verify_fallback_model(self, model):
        """Verify fallback availability and generateContent compatibility."""
        if model in self._verified_fallback_models:
            return
        try:
            model_info = self.sdk_client.models.get(model=model)
        except Exception as error:
            details = _gemini_error_details(error, self._api_key)
            raise GeminiModelError(
                f"Could not verify fallback model {model}: "
                f"{details['error_type']}: {details['message']}",
                status_code=details["status_code"],
                error_type=details["error_type"]
            ) from error

        if isinstance(model_info, dict):
            supported_actions = (
                model_info.get("supported_actions")
                or model_info.get("supported_generation_methods")
                or model_info.get("supportedGenerationMethods")
            )
        else:
            supported_actions = (
                getattr(model_info, "supported_actions", None)
                or getattr(model_info, "supported_generation_methods", None)
                or getattr(model_info, "supportedGenerationMethods", None)
            )
        if not isinstance(supported_actions, (list, tuple, set)) or not any(
            str(action).casefold() == "generatecontent"
            for action in supported_actions
        ):
            raise GeminiModelError(
                f"Fallback model {model} is available but does not advertise "
                "generateContent support."
            )

        # Reject model families for which this client has no known thinking
        # configuration rather than guessing and silently reducing quality.
        _thinking_configuration(model)
        self._verified_fallback_models.add(model)

    def _generate_json_single_model(
        self,
        prompt,
        response_schema,
        retry_on_truncation=False
    ):
        current_prompt = prompt
        response_attempt = 0
        max_response_attempts = 2 if retry_on_truncation else 1
        total_retry_wait = self._total_retry_wait

        while response_attempt < max_response_attempts:
            response_attempt += 1
            transient_attempt = 0
            attempt_output_tokens = (
                self.max_output_tokens
                if response_attempt == 1
                else min(
                    max(
                        self.max_output_tokens * 2,
                        DEFAULT_MAX_OUTPUT_TOKENS
                    ),
                    MAX_MODEL_OUTPUT_TOKENS
                )
            )
            while True:
                transient_attempt += 1
                try:
                    response = self.sdk_client.models.generate_content(
                        model=self.model,
                        contents=current_prompt,
                        config={
                            "max_output_tokens": attempt_output_tokens,
                            "response_mime_type": "application/json",
                            "response_json_schema": response_schema,
                            "thinking_config": _thinking_configuration(self.model),
                            "temperature": 0
                        }
                    )
                    break
                except Exception as error:
                    if (
                        not _is_transient_error(error, self._api_key)
                        or transient_attempt > self.max_retries
                    ):
                        self._raise_safe_error(
                            error,
                            retry_message=(
                                f"Model {self.model} exhausted its bounded "
                                f"retry policy after {transient_attempt} "
                                "request(s). Wait before rerunning; if the "
                                "failure persists, check service status and quota."
                            ) if _is_transient_error(error, self._api_key)
                            else ""
                        )

                    delay = _retry_delay(
                        error,
                        transient_attempt,
                        self.max_retry_wait_seconds - total_retry_wait
                    )
                    if delay is None:
                        retry_message = (
                            "The server's Retry-After delay exceeds the "
                            "configured wait limit. Wait for the indicated "
                            "period before rerunning."
                            if _retry_after_seconds(error) is not None
                            else "The configured retry wait budget is "
                            "exhausted; wait before rerunning."
                        )
                        self._raise_safe_error(
                            error,
                            retry_message=f"Model {self.model}: {retry_message}"
                        )
                    time.sleep(delay)
                    total_retry_wait += delay
                    self._total_retry_wait = total_retry_wait

            try:
                self.last_usage = _response_usage(response)
                return _parse_json_response(
                    response,
                    attempt_output_tokens,
                    attempt=response_attempt
                )
            except GeminiRequestError:
                if (
                    response_attempt >= max_response_attempts
                    or _response_finish_reason(response) != "MAX_TOKENS"
                ):
                    raise
                current_prompt = (
                    f"{prompt}\n\n"
                    "Regenerate exactly one complete JSON object satisfying "
                    "the configured response schema. Keep values concise "
                    "while preserving every required field, supported fact, "
                    "and citation. Do not continue or append to a prior "
                    "response. Do not use Markdown."
                )


    def _raise_safe_error(self, error, retry_message=""):
        details = _gemini_error_details(error, self._api_key)
        status_code = details["status_code"]
        error_type = details["error_type"]
        message = details["message"]
        metadata = {
            "status_code": status_code,
            "error_type": error_type,
            "transient": _is_transient_error(error, self._api_key)
        }
        diagnostic = (
            f"Gemini API error"
            f"{f' HTTP {status_code}' if status_code is not None else ''}"
            f"{f' ({error_type})' if error_type else ''}: {message}"
        )
        if retry_message:
            diagnostic = f"{diagnostic} {retry_message}"

        if status_code == 401:
            raise GeminiAuthenticationError(diagnostic, **metadata) from error
        if status_code == 403:
            raise GeminiPermissionError(diagnostic, **metadata) from error
        if status_code == 429:
            raise GeminiRateLimitError(diagnostic, **metadata) from error
        if status_code == 404:
            raise GeminiModelError(diagnostic, **metadata) from error

        lower_error = f"{error_type or ''} {message}".casefold()
        if status_code == 402 or any(
            phrase in lower_error
            for phrase in (
                "insufficient credit",
                "insufficient_quota",
                "quota exceeded",
                "credit balance",
                "billing",
                "payment required"
            )
        ):
            raise GeminiBillingError(diagnostic, **metadata) from error

        error_name = type(error).__name__
        if any(
            term in error_name.casefold()
            for term in ("connect", "connection", "timeout", "network")
        ):
            raise GeminiRequestError(
                diagnostic,
                error_type=error_type or error_name
            ) from error
        if status_code is not None:
            raise GeminiRequestError(diagnostic, **metadata) from error

        raise error


def _response_finish_reason(response):
    candidates = getattr(response, "candidates", None)
    if not candidates:
        return ""
    reason = getattr(candidates[0], "finish_reason", None)
    if reason is None:
        return ""
    return str(getattr(reason, "value", reason)).rsplit(".", 1)[-1]


def _response_token_count(response):
    usage = getattr(response, "usage_metadata", None)
    tokens = getattr(usage, "candidates_token_count", None)
    if isinstance(tokens, int) and not isinstance(tokens, bool):
        return tokens
    candidates = getattr(response, "candidates", None)
    if candidates:
        count = getattr(candidates[0], "token_count", None)
        if isinstance(count, int) and not isinstance(count, bool):
            return count
    return None


def _response_diagnostic(response, response_chars, max_output_tokens, attempt):
    details = [
        f"attempt={attempt}",
        f"response_chars={response_chars}"
    ]
    finish_reason = _response_finish_reason(response)
    if finish_reason:
        details.append(f"finish_reason={finish_reason}")
    token_count = _response_token_count(response)
    if token_count is not None:
        details.append(f"output_tokens={token_count}")
    usage = getattr(response, "usage_metadata", None)
    for attribute, label in (
        ("prompt_token_count", "prompt_tokens"),
        ("thoughts_token_count", "thoughts_tokens"),
        ("total_token_count", "total_tokens")
    ):
        count = getattr(usage, attribute, None)
        if isinstance(count, int) and not isinstance(count, bool):
            details.append(f"{label}={count}")
    details.append(f"max_output_tokens={max_output_tokens}")
    return ", ".join(details)


def _response_usage(response):
    usage = getattr(response, "usage_metadata", None)
    if usage is None:
        return {}
    fields = {
        "prompt_tokens": "prompt_token_count",
        "completion_tokens": "candidates_token_count",
        "thoughts_tokens": "thoughts_token_count",
        "total_tokens": "total_token_count"
    }
    return {
        label: count
        for label, attribute in fields.items()
        if isinstance(
            count := getattr(usage, attribute, None),
            int
        ) and not isinstance(count, bool)
    }


def _parse_json_response(response, max_output_tokens, attempt=1):
    text = getattr(response, "text", None)
    if not isinstance(text, str) or not text.strip():
        diagnostic = _response_diagnostic(
            response,
            len(text) if isinstance(text, str) else 0,
            max_output_tokens,
            attempt
        )
        raise GeminiRequestError(
            "Gemini returned no structured text content "
            f"({diagnostic})."
        )

    candidate = text.strip()
    fenced_json = JSON_MARKDOWN_FENCE.fullmatch(candidate)
    if fenced_json:
        candidate = fenced_json.group("json").strip()

    try:
        return json.loads(candidate)
    except json.JSONDecodeError as error:
        diagnostic = _response_diagnostic(
            response,
            len(text),
            max_output_tokens,
            attempt
        )
        raise GeminiRequestError(
            "Gemini returned invalid JSON "
            f"(parse_error={error.msg!r}, line={error.lineno}, "
            f"column={error.colno}, position={error.pos}; {diagnostic})."
        ) from error


def _gemini_error_details(error, api_key=""):
    status_code = getattr(error, "code", None)
    if not isinstance(status_code, int):
        status_code = getattr(error, "status_code", None)
    if not isinstance(status_code, int):
        status_code = None

    body = getattr(error, "response_json", None)
    if not isinstance(body, dict):
        body = getattr(error, "body", None)
    if not isinstance(body, dict):
        body = getattr(error, "details", None)

    error_body = body.get("error", body) if isinstance(body, dict) else {}
    if not status_code and isinstance(error_body, dict):
        body_code = error_body.get("code")
        if isinstance(body_code, int):
            status_code = body_code
    error_type = (
        error_body.get("status")
        if isinstance(error_body, dict)
        else None
    )
    if not error_type:
        error_type = getattr(error, "status", None) or type(error).__name__

    message = (
        error_body.get("message")
        if isinstance(error_body, dict)
        else None
    )
    if not message:
        message = getattr(error, "message", None) or str(error)
    if not message:
        message = "No response message was provided."

    return {
        "status_code": status_code,
        "error_type": _redact_sensitive_text(str(error_type), api_key),
        "message": _redact_sensitive_text(str(message), api_key)
    }


def _retry_after_seconds(error):
    response = getattr(error, "response", None)
    headers = getattr(response, "headers", None)
    if headers is None:
        return None

    retry_after = headers.get("Retry-After")
    if not isinstance(retry_after, str):
        return None

    try:
        seconds = int(retry_after.strip())
    except ValueError:
        try:
            retry_at = parsedate_to_datetime(retry_after)
        except (TypeError, ValueError, OverflowError):
            return None
        if retry_at.tzinfo is None:
            retry_at = retry_at.replace(tzinfo=timezone.utc)
        seconds = (retry_at - datetime.now(timezone.utc)).total_seconds()

    if seconds < 0:
        return 0.0
    return float(seconds)


def _is_temporary_rate_limit(error, api_key=""):
    details = _gemini_error_details(error, api_key)
    message = f"{details['error_type']} {details['message']}".casefold()
    if any(term in message for term in (
        "quota exceeded", "daily limit", "per day", "billing",
        "insufficient", "quota exhausted"
    )):
        return False
    if _retry_after_seconds(error) is not None:
        return True
    return any(term in message for term in (
        "rate limit", "rate_limit", "too many requests", "temporar",
        "try again", "retry after", "retry later"
    ))


def _is_transient_error(error, api_key=""):
    status_code = getattr(error, "status_code", None)
    if status_code in TRANSIENT_HTTP_STATUS_CODES and status_code != 429:
        return True
    if status_code == 429:
        return getattr(error, "transient", False) or _is_temporary_rate_limit(
            error, api_key
        )
    status_code = _gemini_error_details(error, api_key)["status_code"]
    if status_code in TRANSIENT_HTTP_STATUS_CODES and status_code != 429:
        return True
    return status_code == 429 and _is_temporary_rate_limit(error, api_key)


def _thinking_configuration(model):
    if re.match(r"^gemini-2\.5(?:\.|-)", model):
        return {"thinking_budget": GEMINI_25_THINKING_BUDGET}
    return {"thinking_level": "LOW"}


def _retry_delay(error, attempt, remaining_wait):
    retry_after = _retry_after_seconds(error)
    if retry_after is None:
        backoff = RETRY_DELAYS_SECONDS[
            min(max(attempt - 1, 0), len(RETRY_DELAYS_SECONDS) - 1)
        ]
        jitter = backoff * RETRY_JITTER_RATIO
        retry_after = random.uniform(backoff - jitter, backoff + jitter)
    else:
        backoff = RETRY_DELAYS_SECONDS[
            min(max(attempt - 1, 0), len(RETRY_DELAYS_SECONDS) - 1)
        ]
        retry_after = max(retry_after, backoff)
    if retry_after > remaining_wait:
        return None
    return retry_after


def _redact_sensitive_text(message, api_key=""):
    if api_key:
        message = message.replace(api_key, "[REDACTED]")
    message = re.sub(
        r"(?i)\bAIza[A-Za-z0-9_-]{20,}\b",
        "[REDACTED]",
        message
    )
    message = re.sub(
        r"(?i)\bBearer\s+\S+",
        "******",
        message
    )
    message = re.sub(
        r"(?im)^\s*(?:authorization|x-goog-api-key|gemini-api-key)\s*:\s*.*$",
        "[REDACTED HEADER]",
        message
    )
    return message[:1000]
