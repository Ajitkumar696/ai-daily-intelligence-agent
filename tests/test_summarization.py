import json

import pytest
import httpx

from src.summarization.cache import SummaryCache, content_fingerprint
from src.summarization.gemini_client import (
    GeminiAuthenticationError,
    GeminiBillingError,
    GeminiClient,
    GeminiConfigurationError,
    GeminiModelError,
    GeminiPermissionError,
    GeminiRateLimitError,
    GeminiRequestError,
    check_gemini_configuration,
    load_gemini_settings
)
from src.summarization.schemas import validate_summary_data
from src.summarization.summarizer import (
    get_max_articles_per_run,
    load_articles,
    save_summaries,
    select_article_content,
    summarize_articles
)
import src.summarize as summarize_cli


def valid_generated_summary(title="Original source title"):
    return {
        "title": title,
        "category": "Research",
        "short_summary": "The source reports a research result.",
        "key_points": ["The report describes a measured result."],
        "why_it_matters": "The result may inform further research.",
        "importance": "Medium",
        "confidence": 0.82
    }


class FakeSummarizer:
    def __init__(self, result=None, error=None):
        self.result = result or valid_generated_summary()
        self.error = error
        self.calls = []

    def generate_summary(self, title, source_content):
        self.calls.append((title, source_content))
        if self.error:
            raise self.error
        result = dict(self.result)
        result["title"] = title
        return result


class FakeMessageResponse:
    def __init__(self, text):
        self.text = text


class FakeModels:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.calls = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return self.response


class FakeSDKClient:
    def __init__(self, response=None, error=None):
        self.models = FakeModels(response=response, error=error)


class SequencedFakeModels:
    def __init__(self, results):
        self.results = list(results)
        self.calls = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


class SequencedFakeSDKClient:
    def __init__(self, results):
        self.models = SequencedFakeModels(results)


def sample_article(**updates):
    article = {
        "source": "OpenAI",
        "title": "Original source title",
        "url": "https://example.com/article",
        "published": "2026-11-02",
        "summary": (
            "The source reports a new research result and explains the "
            "measured performance across several documented experiments."
        )
    }
    article.update(updates)
    return article


def test_gemini_client_returns_valid_structured_mock_response():
    sdk = FakeSDKClient(
        response=FakeMessageResponse(
            json.dumps(valid_generated_summary())
        )
    )
    client = GeminiClient(
        "gemini-test-key",
        model="gemini-3.6-flash",
        max_output_tokens=333,
        sdk_client=sdk
    )

    summary = client.generate_summary(
        "Original source title",
        "A sufficiently long supported source text."
    )

    assert summary == valid_generated_summary()
    request = sdk.models.calls[0]
    assert request["model"] == "gemini-3.6-flash"
    assert request["config"]["max_output_tokens"] == 333
    assert request["config"]["response_mime_type"] == "application/json"
    assert request["config"]["thinking_config"] == {"thinking_level": "LOW"}
    assert request["config"]["thinking_config"] == {"thinking_level": "LOW"}
    response_schema = request["config"]["response_json_schema"]
    assert {
        "category",
        "short_summary",
        "key_points",
        "why_it_matters",
        "importance"
    }.issubset(response_schema["required"])
    assert "title" not in response_schema["properties"]
    assert "response_schema" not in request["config"]
    assert "additional_properties" not in json.dumps(response_schema)
    assert "additionalProperties" not in json.dumps(response_schema)
    assert "Source title: Original source title" in request["contents"]


def test_gemini_25_uses_thinking_budget_instead_of_gemini_3_level():
    sdk = FakeSDKClient(
        response=FakeMessageResponse(json.dumps(valid_generated_summary()))
    )
    client = GeminiClient(
        "gemini-test-key",
        model="gemini-2.5-flash",
        sdk_client=sdk,
        fallback_models=[]
    )

    client.generate_summary("Title", "Content")

    assert sdk.models.calls[0]["config"]["thinking_config"] == {
        "thinking_budget": 8192
    }


def test_gemini_client_rejects_invalid_json():
    client = GeminiClient(
        "gemini-test-key",
        sdk_client=FakeSDKClient(response=FakeMessageResponse("{invalid"))
    )

    with pytest.raises(GeminiRequestError, match="invalid JSON"):
        client.generate_summary("Title", "Content")


def test_gemini_client_rejects_missing_required_fields():
    response = {"title": "Only title"}
    client = GeminiClient(
        "gemini-test-key",
        sdk_client=FakeSDKClient(
            response=FakeMessageResponse(json.dumps(response))
        )
    )

    with pytest.raises(GeminiRequestError, match="missing required fields"):
        client.generate_summary("Title", "Content")


@pytest.mark.parametrize(
    ("status_code", "error_type", "message", "expected_error"),
    [
        (
            401,
            "UNAUTHENTICATED",
            "Invalid API key",
            GeminiAuthenticationError
        ),
        (
            403,
            "PERMISSION_DENIED",
            "Workspace does not have access",
            GeminiPermissionError
        ),
        (
            429,
            "RESOURCE_EXHAUSTED",
            "Rate limit exceeded",
            GeminiRateLimitError
        ),
        (
            400,
            "INVALID_ARGUMENT",
            "Your credit balance is too low",
            GeminiBillingError
        ),
        (
            404,
            "NOT_FOUND",
            "Model does not exist",
            GeminiModelError
        )
    ]
)
def test_gemini_client_preserves_google_api_error_details(
    monkeypatch,
    status_code,
    error_type,
    message,
    expected_error
):
    from google.genai.errors import ClientError
    import src.summarization.gemini_client as gemini_client_module

    monkeypatch.setattr(gemini_client_module.time, "sleep", lambda delay: None)
    sdk_error = ClientError(
        status_code,
        {
            "error": {
                "code": status_code,
                "status": error_type,
                "message": message
            }
        }
    )
    client = GeminiClient(
        "gemini-test-key",
        sdk_client=FakeSDKClient(error=sdk_error)
    )

    with pytest.raises(expected_error) as caught:
        client.generate_summary("Title", "Content")
    assert caught.value.status_code == status_code
    assert caught.value.error_type == error_type
    assert message in str(caught.value)
    expected_calls = 5 if status_code == 429 else 1
    assert len(client.sdk_client.models.calls) == expected_calls


def test_transient_503_retries_once_then_succeeds(monkeypatch):
    from google.genai.errors import ServerError
    import src.summarization.gemini_client as gemini_client_module

    delays = []
    monkeypatch.setattr(gemini_client_module.time, "sleep", delays.append)
    monkeypatch.setattr(gemini_client_module.random, "uniform", lambda low, high: high)
    client = GeminiClient(
        "gemini-test-key",
        sdk_client=SequencedFakeSDKClient([
            ServerError(503, {"error": {
                "status": "UNAVAILABLE",
                "message": "Service temporarily unavailable"
            }}),
            FakeMessageResponse(json.dumps(valid_generated_summary()))
        ])
    )

    result = client.generate_summary("Title", "Content")

    assert result == valid_generated_summary(title="Title")
    assert len(client.sdk_client.models.calls) == 2
    assert delays == [6.0]


def test_transient_503_raises_safe_error_after_bounded_retries(monkeypatch):
    from google.genai.errors import ServerError
    import src.summarization.gemini_client as gemini_client_module

    secret = "gemini-api-key-secret"
    delays = []
    monkeypatch.setattr(gemini_client_module.time, "sleep", delays.append)
    monkeypatch.setattr(gemini_client_module.random, "uniform", lambda low, high: high)
    sdk_error = ServerError(503, {"error": {
        "status": "UNAVAILABLE",
        "message": f"Unavailable; key={secret}"
    }})
    client = GeminiClient(
        secret,
        sdk_client=SequencedFakeSDKClient([sdk_error] * 5)
    )

    with pytest.raises(GeminiRequestError) as caught:
        client.generate_summary("Title", "Content")

    assert caught.value.status_code == 503
    assert caught.value.error_type == "UNAVAILABLE"
    assert "Unavailable" in str(caught.value)
    assert secret not in str(caught.value)
    assert "[REDACTED]" in str(caught.value)
    assert "Wait before rerunning" in str(caught.value)
    assert len(client.sdk_client.models.calls) == 5
    assert delays == [6.0, 18.0, 54.0, 108.0]


@pytest.mark.parametrize(
    ("status_code", "error_type"),
    [
        (429, "RESOURCE_EXHAUSTED"),
        (500, "INTERNAL"),
        (503, "UNAVAILABLE")
    ]
)
def test_transient_http_errors_retry_before_success(
    monkeypatch,
    status_code,
    error_type
):
    from google.genai.errors import ServerError
    import src.summarization.gemini_client as gemini_client_module

    delays = []
    monkeypatch.setattr(gemini_client_module.time, "sleep", delays.append)
    monkeypatch.setattr(gemini_client_module.random, "uniform", lambda low, high: high)
    client = GeminiClient(
        "gemini-test-key",
        sdk_client=SequencedFakeSDKClient([
            ServerError(status_code, {"error": {
                "status": error_type,
                "message": "Temporary service failure"
            }}),
            FakeMessageResponse(json.dumps(valid_generated_summary()))
        ])
    )

    result = client.generate_summary("Title", "Content")

    assert result == valid_generated_summary(title="Title")
    assert len(client.sdk_client.models.calls) == 2
    assert delays == [6.0]


def test_retry_after_header_controls_retry_delay(monkeypatch):
    import httpx
    from google.genai.errors import ServerError
    import src.summarization.gemini_client as gemini_client_module

    delays = []
    monkeypatch.setattr(gemini_client_module.time, "sleep", delays.append)
    monkeypatch.setattr(
        gemini_client_module.random,
        "uniform",
        lambda low, high: pytest.fail("Retry-After must be used as provided.")
    )
    retry_error = ServerError(
        503,
        {"error": {"status": "UNAVAILABLE", "message": "Try later"}},
        response=httpx.Response(503, headers={"Retry-After": "10"})
    )
    client = GeminiClient(
        "gemini-test-key",
        sdk_client=SequencedFakeSDKClient([
            retry_error,
            FakeMessageResponse(json.dumps(valid_generated_summary()))
        ])
    )

    assert client.generate_summary("Title", "Content")
    assert delays == [10.0]


def test_retry_after_beyond_total_wait_budget_stops_without_retry(monkeypatch):
    import httpx
    from google.genai.errors import ServerError
    import src.summarization.gemini_client as gemini_client_module

    monkeypatch.setattr(
        gemini_client_module.time,
        "sleep",
        lambda delay: pytest.fail("The retry wait budget must not be exceeded.")
    )
    retry_error = ServerError(
        503,
        {"error": {"status": "UNAVAILABLE", "message": "Try later"}},
        response=httpx.Response(503, headers={"Retry-After": "301"})
    )
    client = GeminiClient(
        "gemini-test-key",
        sdk_client=SequencedFakeSDKClient([retry_error])
    )

    with pytest.raises(GeminiRequestError) as caught:
        client.generate_summary("Title", "Content")

    assert caught.value.status_code == 503
    assert len(client.sdk_client.models.calls) == 1


def test_transient_retry_uses_jitter_within_exponential_backoff(monkeypatch):
    from google.genai.errors import ServerError
    import src.summarization.gemini_client as gemini_client_module

    delays = []
    monkeypatch.setattr(gemini_client_module.time, "sleep", delays.append)
    monkeypatch.setattr(
        gemini_client_module.random,
        "uniform",
        lambda low, high: (low + high) / 2
    )
    client = GeminiClient(
        "gemini-test-key",
        sdk_client=SequencedFakeSDKClient([
            ServerError(503, {"error": {
                "status": "UNAVAILABLE",
                "message": "Temporarily unavailable"
            }}),
            FakeMessageResponse(json.dumps(valid_generated_summary()))
        ])
    )

    assert client.generate_summary("Title", "Content")
    assert delays == [5.0]
    assert len(client.sdk_client.models.calls) == 2


@pytest.mark.parametrize(
    ("status_code", "message"),
    [(429, "Daily quota exceeded"), (502, "Bad gateway")]
)
def test_non_temporary_rate_limit_and_unsupported_5xx_are_not_retried(
    monkeypatch,
    status_code,
    message
):
    from google.genai.errors import ServerError
    import src.summarization.gemini_client as gemini_client_module

    monkeypatch.setattr(
        gemini_client_module.time,
        "sleep",
        lambda delay: pytest.fail("This response must not be retried.")
    )
    client = GeminiClient(
        "gemini-test-key",
        sdk_client=SequencedFakeSDKClient([
            ServerError(status_code, {"error": {
                "status": "RESOURCE_EXHAUSTED" if status_code == 429 else "BAD_GATEWAY",
                "message": message
            }})
        ])
    )

    with pytest.raises(GeminiRequestError):
        client.generate_summary("Title", "Content")
    assert len(client.sdk_client.models.calls) == 1


def test_gemini_retry_limits_are_configurable(monkeypatch):
    from google.genai.errors import ServerError
    import src.summarization.gemini_client as gemini_client_module

    monkeypatch.setenv("GEMINI_MAX_RETRIES", "2")
    monkeypatch.setenv("GEMINI_MAX_RETRY_WAIT_SECONDS", "75")
    delays = []
    monkeypatch.setattr(gemini_client_module.time, "sleep", delays.append)
    monkeypatch.setattr(gemini_client_module.random, "uniform", lambda low, high: high)

    error = ServerError(503, {"error": {
        "status": "UNAVAILABLE",
        "message": "Temporarily unavailable"
    }})
    client = GeminiClient(
        "gemini-test-key",
        sdk_client=FakeSDKClient(error=error)
    )

    assert client.max_retries == 2
    assert client.max_retry_wait_seconds == 75
    with pytest.raises(GeminiRequestError):
        client.generate_summary("Title", "Content")
    assert len(client.sdk_client.models.calls) == 3
    assert delays == [6.0, 18.0]


def test_ordered_gemini_model_fallbacks_are_configurable(monkeypatch):
    from src.summarization.gemini_client import load_gemini_model_fallbacks

    monkeypatch.setenv(
        "GEMINI_MODEL_FALLBACKS",
        "gemini-3.6-flash,gemini-2.5-flash"
    )

    assert load_gemini_model_fallbacks("gemini-3.8-flash") == [
        "gemini-3.6-flash",
        "gemini-2.5-flash"
    ]


def test_temporary_429_respects_retry_after(monkeypatch):
    import httpx
    from google.genai.errors import ServerError
    import src.summarization.gemini_client as gemini_client_module

    delays = []
    monkeypatch.setattr(gemini_client_module.time, "sleep", delays.append)
    monkeypatch.setattr(
        gemini_client_module.random,
        "uniform",
        lambda low, high: pytest.fail("Retry-After should determine the wait.")
    )
    error = ServerError(
        429,
        {"error": {"status": "RESOURCE_EXHAUSTED", "message": "Rate limit exceeded"}},
        response=httpx.Response(429, headers={"Retry-After": "12"})
    )
    client = GeminiClient(
        "gemini-test-key",
        sdk_client=SequencedFakeSDKClient([
            error,
            FakeMessageResponse(json.dumps(valid_generated_summary()))
        ])
    )

    assert client.generate_summary("Title", "Content")
    assert delays == [12.0]


def test_permanent_error_is_not_retried(monkeypatch):
    from google.genai.errors import ClientError
    import src.summarization.gemini_client as gemini_client_module

    monkeypatch.setattr(
        gemini_client_module.time,
        "sleep",
        lambda delay: pytest.fail("Permanent errors must not be retried.")
    )
    client = GeminiClient(
        "gemini-test-key",
        sdk_client=SequencedFakeSDKClient([
            ClientError(400, {"error": {
                "status": "INVALID_ARGUMENT",
                "message": "Invalid request"
            }})
        ])
    )

    with pytest.raises(GeminiRequestError) as caught:
        client.generate_summary("Title", "Content")

    assert caught.value.status_code == 400
    assert len(client.sdk_client.models.calls) == 1


def test_network_failure_is_classified_without_http_status():
    client = GeminiClient(
        "gemini-test-key",
        sdk_client=FakeSDKClient(
            error=httpx.ConnectError("Network is unreachable")
        )
    )

    with pytest.raises(GeminiRequestError) as caught:
        client.generate_summary("Title", "Content")
    assert caught.value.status_code is None
    assert caught.value.error_type == "ConnectError"
    assert "Network is unreachable" in str(caught.value)


def test_error_message_redacts_credentials_and_request_headers():
    from src.summarization.gemini_client import _redact_sensitive_text

    message = _redact_sensitive_text(
        "Invalid key AIzaSyPRIVATE_KEY_1234567890ABCDEF\n"
        "Authorization: Bearer private-token\n"
        "x-goog-api-key: private-header-value"
    )

    assert "AIzaSyPRIVATE_KEY_1234567890ABCDEF" not in message
    assert "private-header-value" not in message
    assert message.count("[REDACTED") == 3


def test_gemini_client_does_not_hide_unexpected_sdk_errors():
    client = GeminiClient(
        "gemini-test-key",
        model="gemini-3.6-flash",
        sdk_client=FakeSDKClient(
            error=RuntimeError("unexpected SDK programming error")
        )
    )

    with pytest.raises(RuntimeError, match="unexpected SDK programming error"):
        client.generate_summary("Title", "Content")


def test_content_selection_prefers_extracted_text_then_summary_then_metadata():
    full_text = "Extracted full text with enough context."
    article = sample_article(
        extracted_content={"full_text": full_text},
        description="Fallback description"
    )
    assert select_article_content(article) == (
        full_text,
        "extracted_full_text"
    )

    article.pop("extracted_content")
    assert select_article_content(article) == (
        article["summary"],
        "source_summary"
    )

    article["summary"] = ""
    assert select_article_content(article) == (
        "Fallback description",
        "descriptive_metadata"
    )


def test_insufficient_content_does_not_call_gemini(tmp_path):
    client = FakeSummarizer()
    article = sample_article(summary="A release happened.")

    result = summarize_articles(
        [article],
        client=client,
        cache=SummaryCache(tmp_path / "cache.json")
    )

    assert result[0]["summary_status"] == "INSUFFICIENT_CONTENT"
    assert client.calls == []


def test_success_summary_has_required_fields_and_preserves_source_data(tmp_path):
    article = sample_article()
    client = FakeSummarizer()
    cache = SummaryCache(tmp_path / "cache.json")

    result = summarize_articles([article], client=client, cache=cache)

    expected_fields = {
        "title",
        "source",
        "source_url",
        "published_date",
        "category",
        "short_summary",
        "key_points",
        "why_it_matters",
        "importance",
        "confidence",
        "summary_status",
        "content_source_type"
    }
    assert expected_fields.issubset(result[0])
    assert result[0]["summary_status"] == "SUCCESS"
    assert result[0]["title"] == article["title"]
    assert result[0]["source"] == article["source"]
    assert result[0]["source_url"] == article["url"]
    assert result[0]["published_date"] == article["published"]
    assert validate_summary_data({
        field: result[0][field]
        for field in valid_generated_summary()
    })


def test_api_failure_is_preserved_and_api_key_is_redacted(monkeypatch, tmp_path):
    secret = "private-gemini-key-value"
    monkeypatch.setenv("GEMINI_API_KEY", secret)
    client = FakeSummarizer(
        error=GeminiRequestError(f"failure with {secret}")
    )

    result = summarize_articles(
        [sample_article()],
        client=client,
        cache=SummaryCache(tmp_path / "cache.json")
    )

    assert result[0]["summary_status"] == "FAILED"
    assert "[REDACTED]" in result[0]["error"]
    assert secret not in result[0]["error"]


def test_summary_cache_hit_avoids_api_and_changed_content_misses(
    tmp_path,
    capsys
):
    article = sample_article()
    cache = SummaryCache(tmp_path / "cache.json")
    first_client = FakeSummarizer()

    first = summarize_articles([article], client=first_client, cache=cache)
    assert "API calls=1" in capsys.readouterr().out
    second_client = FakeSummarizer()
    second = summarize_articles([article], client=second_client, cache=cache)
    assert "API calls=0, successful cache hits=1" in capsys.readouterr().out

    assert first[0]["summary_status"] == "SUCCESS"
    assert second[0]["summary_status"] == "SUCCESS"
    assert second_client.calls == []

    changed = sample_article(summary=article["summary"] + " Additional fact.")
    third_client = FakeSummarizer()
    third = summarize_articles([changed], client=third_client, cache=cache)
    assert "API calls=1" in capsys.readouterr().out
    assert third[0]["summary_status"] == "SUCCESS"
    assert len(third_client.calls) == 1
    assert content_fingerprint(
        article["title"],
        article["summary"]
    ) != content_fingerprint(article["title"], changed["summary"])


def test_maximum_article_limit_and_no_live_client_creation(monkeypatch, tmp_path):
    import src.summarization.summarizer as summarizer_module

    client = FakeSummarizer()
    articles = [
        sample_article(url=f"https://example.com/{index}")
        for index in range(5)
    ]
    monkeypatch.setattr(
        summarizer_module,
        "create_llm_client",
        lambda: pytest.fail("Tests must inject a mocked Gemini client.")
    )

    results = summarize_articles(
        articles,
        client=client,
        cache=SummaryCache(tmp_path / "cache.json"),
        max_articles=3
    )

    assert len(results) == 3
    assert len(client.calls) == 3


def test_configured_maximum_defaults_to_three_and_supports_all(
    monkeypatch,
    tmp_path
):
    import src.summarization.summarizer as summarizer_module

    monkeypatch.delenv("MAX_ARTICLES_PER_RUN", raising=False)
    monkeypatch.setattr(summarizer_module, "PROJECT_ROOT", tmp_path)
    assert get_max_articles_per_run() == 3
    assert get_max_articles_per_run("5") == 5
    assert get_max_articles_per_run("0") == 0
    with pytest.raises(ValueError, match="positive integer"):
        get_max_articles_per_run("-1")


def test_missing_api_key_marks_record_failed_without_live_request(
    monkeypatch,
    tmp_path
):
    import src.summarization.gemini_client as gemini_client_module
    import src.summarization.summarizer as summarizer_module
    import src.summarization.providers as providers_module

    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_BASE_URL", raising=False)
    monkeypatch.setattr(providers_module, "load_dotenv", lambda **_kwargs: False)
    monkeypatch.setattr(summarizer_module, "PROJECT_ROOT", tmp_path)
    # The environment has no credential. Client construction must fail before
    # it can instantiate the SDK or contact the API.
    with pytest.raises(
        GeminiConfigurationError,
        match="GROQ_API_KEY"
    ):
        summarize_articles(
            [sample_article()],
            cache=SummaryCache(tmp_path / "cache.json")
        )


def test_root_dotenv_and_valid_key_are_loaded_without_api_request(
    monkeypatch,
    tmp_path
):
    import src.summarization.gemini_client as gemini_client_module

    (tmp_path / ".env").write_text(
        "GEMINI_API_KEY=gemini-mocked-key\n"
        "GEMINI_MODEL=gemini-3.8-flash\n"
        "GEMINI_MAX_OUTPUT_TOKENS=456\n",
        encoding="utf-8"
    )
    monkeypatch.setattr(gemini_client_module, "PROJECT_ROOT", tmp_path)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    monkeypatch.delenv("GEMINI_MAX_OUTPUT_TOKENS", raising=False)
    monkeypatch.setattr(
        gemini_client_module,
        "find_spec",
        lambda name: object() if name == "google.genai" else None
    )

    status = check_gemini_configuration()

    assert status == {
        "ready": True,
        "api_key_configured": True,
        "model": "gemini-3.8-flash",
        "max_output_tokens": 456,
        "google_genai_sdk_installed": True,
        "issues": []
    }
    assert load_gemini_settings() == (
        "gemini-mocked-key",
        "gemini-3.8-flash",
        456
    )


def test_process_environment_key_precedes_dotenv_and_sdk_receives_same_key(
    monkeypatch,
    tmp_path
):
    import src.summarization.gemini_client as gemini_client_module

    (tmp_path / ".env").write_text(
        "GEMINI_API_KEY=dotenv-gemini-key\n",
        encoding="utf-8"
    )
    constructed = {}
    from google import genai

    def fake_gemini_client(**kwargs):
        constructed.update(kwargs)
        return FakeSDKClient()

    monkeypatch.setattr(gemini_client_module, "PROJECT_ROOT", tmp_path)
    monkeypatch.setenv("GEMINI_API_KEY", "  process-gemini-key  ")
    monkeypatch.setattr(genai, "Client", fake_gemini_client)

    diagnostic_settings = load_gemini_settings()
    client = GeminiClient.from_environment()

    assert diagnostic_settings[0] == "process-gemini-key"
    assert constructed["api_key"] == diagnostic_settings[0]
    assert constructed["http_options"].retry_options.attempts == 1
    assert client.sdk_client is not None


@pytest.mark.parametrize(
    ("status_code", "error_type", "message"),
    [
        (401, "UNAUTHENTICATED", "Invalid API key"),
        (403, "PERMISSION_DENIED", "Access denied"),
        (429, "RESOURCE_EXHAUSTED", "Rate limit exceeded"),
        (400, "INVALID_ARGUMENT", "Your credit balance is too low"),
        (404, "NOT_FOUND", "Model does not exist")
    ]
)
def test_api_error_details_are_written_to_summary_json(
    tmp_path,
    capsys,
    monkeypatch,
    status_code,
    error_type,
    message
):
    from src.summarization.summarizer import save_summaries
    import src.summarization.gemini_client as gemini_client_module
    from google.genai.errors import ClientError

    monkeypatch.setattr(gemini_client_module.time, "sleep", lambda delay: None)
    sdk_error = ClientError(
        status_code,
        {
            "error": {
                "code": status_code,
                "status": error_type,
                "message": message
            }
        }
    )
    client = GeminiClient(
        "gemini-test-key",
        sdk_client=FakeSDKClient(error=sdk_error)
    )
    result = summarize_articles(
        [sample_article()],
        client=client,
        cache=SummaryCache(tmp_path / "cache.json")
    )
    output_file = tmp_path / "summaries.json"
    save_summaries(result, output_file)
    saved = json.loads(output_file.read_text(encoding="utf-8"))[0]

    assert saved["summary_status"] == "FAILED"
    assert saved["error_status_code"] == status_code
    assert saved["error_type"] == error_type
    assert message in saved["error"]
    expected_calls = 5 if status_code == 429 else 1
    assert len(client.sdk_client.models.calls) == expected_calls
    assert "API calls=1" in capsys.readouterr().out


def test_authentication_response_never_persists_api_key(tmp_path):
    from src.summarization.summarizer import save_summaries
    from google.genai.errors import ClientError

    secret = "private-gemini-key-value"
    client = GeminiClient(
        secret,
        sdk_client=FakeSDKClient(
            error=ClientError(
                401,
                {
                    "error": {
                        "code": 401,
                        "status": "UNAUTHENTICATED",
                        "message": f"Invalid key {secret}"
                    }
                }
            )
        )
    )
    result = summarize_articles(
        [sample_article()],
        client=client,
        cache=SummaryCache(tmp_path / "cache.json")
    )
    output_file = tmp_path / "summaries.json"
    save_summaries(result, output_file)
    serialized = output_file.read_text(encoding="utf-8")

    assert secret not in serialized
    record = json.loads(serialized)[0]
    assert record["error_status_code"] == 401
    assert record["error_type"] == "UNAUTHENTICATED"
    assert "[REDACTED]" in record["error"]


def test_client_initializes_official_google_sdk_without_request(
    monkeypatch,
    tmp_path
):
    import src.summarization.gemini_client as gemini_client_module
    from google import genai

    constructed = {}

    def fake_gemini(**kwargs):
        constructed.update(kwargs)
        return FakeSDKClient()

    monkeypatch.setattr(gemini_client_module, "PROJECT_ROOT", tmp_path)
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-mocked-key")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-3.8-flash")
    monkeypatch.setenv("GEMINI_MAX_OUTPUT_TOKENS", "456")
    monkeypatch.setattr(genai, "Client", fake_gemini)

    client = GeminiClient.from_environment()

    assert client.model == "gemini-3.8-flash"
    assert client.max_output_tokens == 456
    assert constructed["api_key"] == "gemini-mocked-key"
    assert constructed["http_options"].retry_options.attempts == 1


def test_safe_cli_diagnostic_makes_no_api_request(
    monkeypatch,
    tmp_path,
    capsys
):
    import src.summarization.summarizer as summarizer_module
    import src.summarization.providers as providers_module

    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_BASE_URL", raising=False)
    monkeypatch.setattr(providers_module, "load_dotenv", lambda **_kwargs: False)
    monkeypatch.setattr(
        summarizer_module,
        "create_llm_client",
        lambda: pytest.fail("The diagnostic must not create an API client.")
    )

    assert summarize_cli.main(["--check-config"]) == 1
    output = capsys.readouterr().out
    assert "GROQ_API_KEY" in output
    assert "Groq configuration invalid" in output
    assert "AIza" not in output
    assert "gemini-mocked-key" not in output


@pytest.mark.parametrize(
    ("setting", "value", "message"),
    [
        ("GROQ_API_KEY", "your_groq_api_key", "GROQ_API_KEY"),
        ("GROQ_BASE_URL", "http://api.groq.com/openai/v1", "GROQ_BASE_URL"),
        ("GROQ_MODEL", "invalid model", "GROQ_MODEL"),
    ]
)
def test_invalid_groq_configuration_fails_before_client_creation(
    monkeypatch,
    tmp_path,
    setting,
    value,
    message,
):
    import src.summarization.summarizer as summarizer_module
    import src.summarization.providers as providers_module

    monkeypatch.setattr(summarizer_module, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(providers_module, "load_dotenv", lambda **_kwargs: False)
    monkeypatch.setenv("GROQ_API_KEY", "groq-test-key")
    monkeypatch.setenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
    monkeypatch.setenv("GROQ_MODEL", "openai/gpt-oss-20b")
    monkeypatch.setenv(setting, value)

    with pytest.raises(GeminiConfigurationError, match=message):
        summarize_articles(
            [sample_article()],
            cache=SummaryCache(tmp_path / "cache.json")
        )


def test_unexpected_client_exception_is_not_converted_to_article_failure(
    tmp_path
):
    client = FakeSummarizer(error=RuntimeError("unexpected programming error"))

    with pytest.raises(RuntimeError, match="unexpected programming error"):
        summarize_articles(
            [sample_article()],
            client=client,
            cache=SummaryCache(tmp_path / "cache.json")
        )


def test_load_and_save_summary_json_use_requested_temp_paths(tmp_path):
    input_path = tmp_path / "articles.json"
    output_path = tmp_path / "summaries.json"
    input_path.write_text(
        json.dumps([sample_article()]),
        encoding="utf-8"
    )

    articles = load_articles(input_path)
    client = FakeSummarizer()
    summaries = summarize_articles(
        articles,
        client=client,
        cache=SummaryCache(tmp_path / "cache.json")
    )
    saved = save_summaries(summaries, output_path)

    assert saved == output_path
    loaded = json.loads(output_path.read_text(encoding="utf-8"))
    assert loaded[0]["summary_status"] == "SUCCESS"
    assert loaded[0]["source_url"] == "https://example.com/article"
