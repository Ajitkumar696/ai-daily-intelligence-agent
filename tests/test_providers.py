import json
from pathlib import Path

import pytest

from src.summarization.providers import (
    GROQ_DEFAULT_BASE_URL,
    GROQ_DEFAULT_MODEL,
    GroqClient,
    GroqProviderError,
    _strict_schema,
    check_groq_configuration,
    create_llm_client,
)


class Response:
    def __init__(self, status=200, content='{"ok":true}', data=None):
        self.status_code = status
        self.ok = 200 <= status < 300
        self.content = content
        self.data = data

    def json(self):
        if self.data is not None:
            return self.data
        return {
            "choices": [{"message": {"content": self.content}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
        }


class Session:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


@pytest.fixture(autouse=True)
def no_dotenv(monkeypatch, tmp_path):
    import src.summarization.providers as providers
    import src.summarization.token_budget as token_budget
    monkeypatch.setattr(providers, "load_dotenv", lambda **kwargs: False)
    monkeypatch.setattr(
        token_budget, "DEFAULT_USAGE_FILE", tmp_path / "groq-token-usage.json"
    )


def test_groq_success_uses_json_schema_and_local_schema_validation():
    from src.research.daily_programme import DAILY_PROGRAMME_SCHEMA
    session = Session([Response()])
    client = GroqClient("test-key", "openai/gpt-oss-20b", session=session)

    assert client.generate_daily_programme("grounded prompt") == {"ok": True}
    url, request = session.calls[0]
    assert url == GROQ_DEFAULT_BASE_URL + "/chat/completions"
    assert request["headers"]["Authorization"] == "Bearer test-key"
    schema = request["json"]["response_format"]["json_schema"]
    assert schema["strict"] is False
    assert schema["schema"] == _strict_schema(DAILY_PROGRAMME_SCHEMA)
    assert client.last_usage["total_tokens"] == 8


def test_story_batch_uses_strict_json_schema_for_supported_model():
    from src.research.daily_programme import DAILY_STORY_BATCH_SCHEMA

    session = Session([Response(content='{"stories":[]}')])
    client = GroqClient("test-key", session=session, max_retries=0)

    assert client.generate_daily_story_batch("batch prompt") == {"stories": []}
    _, request = session.calls[0]
    response_format = request["json"]["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["strict"] is True
    assert response_format["json_schema"]["schema"] == _strict_schema(
        DAILY_STORY_BATCH_SCHEMA
    )
    messages = request["json"]["messages"]
    assert client.last_stage == "story_batch"


def test_story_batch_uses_json_object_mode_only_for_unsupported_model():
    session = Session([Response(content='{"stories":[]}')])
    client = GroqClient(
        "test-key", "unsupported/model", session=session, max_retries=0
    )

    assert client.generate_daily_story_batch("batch prompt") == {"stories": []}
    _, request = session.calls[0]
    assert request["json"]["response_format"] == {"type": "json_object"}
    assert "json" in request["json"]["messages"][0]["content"].casefold()


def test_final_script_keeps_strict_json_schema_mode_and_records_stage():
    from src.research.daily_programme import DAILY_PROGRAMME_SCRIPTS_SCHEMA

    session = Session([Response()])
    client = GroqClient("test-key", session=session, max_retries=0)

    client.generate_programme_scripts("script prompt", DAILY_PROGRAMME_SCRIPTS_SCHEMA)
    _, request = session.calls[0]
    response_format = request["json"]["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["strict"] is True
    assert response_format["json_schema"]["schema"] == _strict_schema(
        DAILY_PROGRAMME_SCRIPTS_SCHEMA
    )
    assert client.last_stage == "final_script"


def test_summary_request_is_stage_attributed():
    from src.summarization.schemas import CATEGORIES

    result = {
        "title": "A source title",
        "category": CATEGORIES[0],
        "short_summary": "A concise source-grounded summary.",
        "key_points": ["A supported source point."],
        "why_it_matters": "The source describes a relevant development.",
        "importance": "Low",
        "confidence": 0.7,
    }
    session = Session([Response(content=json.dumps(result))])
    client = GroqClient("test-key", session=session, max_retries=0)

    client.generate_summary("A source title", "The provided source content.")
    assert client.last_stage == "summary"
    assert session.calls[0][1]["json"]["response_format"]["json_schema"][
        "strict"
    ] is True


def test_invalid_api_response_and_json_are_actionable():
    for response, message in [
        (Response(data={"unexpected": True}), "invalid API response"),
        (Response(content="not json"), "invalid JSON"),
    ]:
        client = GroqClient("test-key", session=Session([response]), max_retries=0)
        with pytest.raises(GroqProviderError, match=message):
            client.generate_daily_programme("prompt")


def test_invalid_story_batch_json_has_stage_attribution():
    client = GroqClient(
        "test-key", session=Session([Response(content="not json")]),
        max_retries=0,
    )

    with pytest.raises(GroqProviderError, match="story_batch.*invalid JSON") as caught:
        client.generate_daily_story_batch("batch prompt")

    assert caught.value.stage == "story_batch"
    assert caught.value.error_type == "INVALID_JSON"


@pytest.mark.parametrize(
    ("content", "reported_type"),
    [
        ("[]", "array"),
        ("[1, 2]", "array"),
        ('[{"a": 1}, "x"]', "array"),
        ('"unexpected"', "string"),
        ("null", "null"),
    ],
)
def test_story_batch_rejects_non_object_json_with_safe_type_diagnostic(
    content, reported_type
):
    client = GroqClient(
        "test-key", session=Session([Response(content=content)]),
        max_retries=0,
    )

    with pytest.raises(GroqProviderError) as caught:
        client.generate_daily_story_batch("private source packet")

    assert caught.value.stage == "story_batch"
    assert caught.value.error_type == "INVALID_API_RESPONSE"
    assert f"top-level type={reported_type}" in str(caught.value)
    assert "private source packet" not in str(caught.value)


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        # bare array of research objects is wrapped
        ('[{"summary": "a"}, {"summary": "b"}]',
         {"stories": [{"summary": "a"}, {"summary": "b"}]}),
        # array holding the {"stories": [...]} wrapper is unwrapped
        ('[{"stories": [{"summary": "a"}]}]', {"stories": [{"summary": "a"}]}),
        # the documented shape is untouched
        ('{"stories": [{"summary": "a"}]}', {"stories": [{"summary": "a"}]}),
    ],
)
def test_story_batch_accepts_bare_array_of_research_objects(content, expected):
    client = GroqClient(
        "test-key", session=Session([Response(content=content)]),
        max_retries=0,
    )

    assert client.generate_daily_story_batch("batch prompt") == expected


def test_array_normalization_is_limited_to_the_story_batch_stage():
    client = GroqClient(
        "test-key", session=Session([Response(content='[{"a": 1}]')]),
        max_retries=0,
    )

    with pytest.raises(GroqProviderError, match="top-level type=array"):
        client.generate_daily_programme("prompt")


def test_story_batch_error_includes_batch_context_without_source_content():
    from datetime import date

    from src.research.daily_programme import (
        _generate_batched_daily_programme,
        _prepare_groups,
    )

    client = GroqClient(
        "test-key", session=Session([Response(content="[]")]),
        max_retries=0,
    )
    groups = _prepare_groups([{
        "topic": "Trusted topic",
        "sources": [{
            "publisher": "Trusted publisher",
            "title": "Trusted article title",
            "url": "https://example.test/article",
            "content": "private article text",
        }],
    }])

    with pytest.raises(GroqProviderError) as caught:
        _generate_batched_daily_programme(groups, date(2026, 11, 5), client)

    assert "stage=story_batch" in str(caught.value) or caught.value.stage == "story_batch"
    assert "batch=1/1" in str(caught.value)
    assert "top-level type=array" in str(caught.value)
    assert "private article text" not in str(caught.value)


def test_rate_limit_and_server_errors_retry_with_bounded_backoff(monkeypatch):
    import src.summarization.providers as providers
    delays = []
    monkeypatch.setattr(providers.time, "sleep", delays.append)
    session = Session([Response(429), Response(503), Response()])
    client = GroqClient("test-key", session=session, max_retries=2, retry_delays=(0.1, 0.2))

    assert client.generate_daily_programme("prompt") == {"ok": True}
    assert len(session.calls) == 3
    assert delays == [0.1, 0.2]


def test_permanent_error_does_not_retry_and_suggests_configuration(monkeypatch):
    import src.summarization.providers as providers
    delays = []
    monkeypatch.setattr(providers.time, "sleep", delays.append)
    client = GroqClient("test-key", session=Session([Response(401)]))

    with pytest.raises(GroqProviderError, match="GROQ_API_KEY") as caught:
        client.generate_daily_programme("prompt")
    assert caught.value.status_code == 401
    assert delays == []


@pytest.mark.parametrize(
    ("status", "detail"),
    [
        (400, "invalid_request_error: schema_validation: unsupported schema"),
        (413, "request_too_large: request exceeds the configured limit"),
    ],
)
def test_http_request_errors_include_only_sanitized_provider_details(
    status, detail
):
    secret = "private-groq-test-secret"
    response = Response(
        status=status,
        data={"error": {
            "type": "invalid_request_error",
            "code": "schema_validation",
            "message": f"{detail}; credential={secret}",
        }},
    )
    client = GroqClient(secret, session=Session([response]), max_retries=0)

    with pytest.raises(GroqProviderError) as caught:
        client.generate_daily_programme("prompt")
    assert caught.value.status_code == status
    assert "API detail:" in str(caught.value)
    assert "[REDACTED]" in str(caught.value)
    assert secret not in str(caught.value)


def test_http_errors_preserve_bounded_redacted_failed_generation_and_stage():
    secret = "private-groq-test-secret"
    failed_generation = (
        "Generated fragment with Bearer private-token and "
        "gsk_1234567890abcdefghijklmnopqrstuvwxyz. "
        + ("x" * 1000)
    )
    response = Response(
        status=400,
        data={"error": {
            "type": "invalid_request_error",
            "code": "json_validate_failed",
            "message": f"Generation rejected; key={secret}",
            "failed_generation": failed_generation,
        }},
    )
    client = GroqClient(secret, session=Session([response]), max_retries=0)

    with pytest.raises(GroqProviderError) as caught:
        client.generate_daily_story_batch("batch prompt")

    error = caught.value
    assert error.stage == "story_batch"
    assert error.api_detail.startswith("type=invalid_request_error")
    assert "code=json_validate_failed" in error.api_detail
    assert "failed_generation=" in error.api_detail
    assert "[REDACTED]" in error.api_detail
    assert "private-token" not in error.api_detail
    assert "gsk_1234567890" not in error.api_detail
    assert secret not in error.api_detail
    assert len(error.api_detail) <= 900


def test_groq_is_the_only_factory_provider_and_reads_groq_configuration(monkeypatch):
    import src.summarization.providers as providers
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setenv("GROQ_BASE_URL", GROQ_DEFAULT_BASE_URL)
    monkeypatch.setenv("GROQ_MODEL", GROQ_DEFAULT_MODEL)

    assert isinstance(create_llm_client(), GroqClient)
    assert check_groq_configuration() == {
        "ready": True,
        "provider": "groq",
        "model": GROQ_DEFAULT_MODEL,
        "base_url": GROQ_DEFAULT_BASE_URL,
    }
    assert providers.GROQ_DEFAULT_MODEL == "openai/gpt-oss-20b"


def _rate_limit_response(message):
    return Response(status=429, data={"error": {
        "type": "tokens", "code": "rate_limit_exceeded", "message": message,
    }})


def test_rate_limit_hint_sets_the_wait_instead_of_the_fixed_delay(monkeypatch):
    import src.summarization.providers as providers
    delays = []
    monkeypatch.setattr(providers.time, "sleep", delays.append)
    session = Session([
        _rate_limit_response("Limit 8000, Used 6089. Please try again in 44.085s."),
        Response(),
    ])
    client = GroqClient("test-key", session=session, max_retries=2)

    assert client.generate_daily_programme("prompt") == {"ok": True}
    assert delays == [pytest.approx(45.085)]


def test_rate_limit_hint_never_shortens_the_default_delay(monkeypatch):
    import src.summarization.providers as providers
    delays = []
    monkeypatch.setattr(providers.time, "sleep", delays.append)
    session = Session([_rate_limit_response("Please try again in 0.5s."), Response()])
    client = GroqClient("test-key", session=session, max_retries=2)

    client.generate_daily_programme("prompt")
    assert delays == [5.0]


def test_long_rate_limit_wait_is_reported_instead_of_slept(monkeypatch):
    import src.summarization.providers as providers
    delays = []
    monkeypatch.setattr(providers.time, "sleep", delays.append)
    session = Session([_rate_limit_response("Please try again in 7m12.5s.")])
    client = GroqClient("test-key", session=session, max_retries=3)

    with pytest.raises(GroqProviderError, match="HTTP 429") as caught:
        client.generate_daily_programme("prompt")
    assert delays == []
    assert caught.value.status_code == 429
    assert len(session.calls) == 1


def test_retry_after_header_is_honoured(monkeypatch):
    import src.summarization.providers as providers
    delays = []
    monkeypatch.setattr(providers.time, "sleep", delays.append)
    limited = Response(status=429)
    limited.headers = {"retry-after": "30"}
    client = GroqClient("test-key", session=Session([limited, Response()]), max_retries=2)

    client.generate_daily_programme("prompt")
    assert delays == [31.0]


def test_daily_token_quota_stops_without_retry_and_persists_across_clients(tmp_path):
    from src.summarization.token_budget import GroqTokenBudget

    response = Response(status=429, data={"error": {
        "type": "tokens",
        "code": "rate_limit_exceeded",
        "message": (
            "Rate limit reached on tokens per day (TPD): Limit 200000, "
            "Used 197271, Requested 4397. Please try again in 6h."
        ),
    }})
    response.headers = {}
    budget = GroqTokenBudget(path=tmp_path / "usage.json")
    session = Session([response])
    client = GroqClient(
        "test-key", session=session, max_retries=4, token_budget=budget
    )

    with pytest.raises(GroqProviderError) as caught:
        client.generate_daily_story_batch("batch prompt")
    assert caught.value.error_type == "DAILY_QUOTA_EXHAUSTED"
    assert len(session.calls) == 1
    assert budget.used(GROQ_DEFAULT_MODEL) == 197271

    next_session = Session([])
    next_client = GroqClient(
        "test-key", session=next_session,
        token_budget=GroqTokenBudget(path=tmp_path / "usage.json"),
    )
    with pytest.raises(GroqProviderError) as blocked:
        next_client.generate_summary("title", "source content")
    assert blocked.value.error_type == "TOKEN_BUDGET_EXHAUSTED"
    assert next_session.calls == []


def test_daily_budget_preflights_every_stage_and_reserves_final_script(tmp_path):
    from src.summarization.token_budget import GroqTokenBudget

    budget = GroqTokenBudget(
        path=tmp_path / "usage.json", daily_limit=2500,
        final_script_reserve=1200,
    )
    session = Session([Response(), Response()])
    client = GroqClient(
        "test-key", session=session, max_retries=0,
        max_output_tokens=100, token_budget=budget,
    )
    client.generate_daily_story_batch("a small prompt")
    assert len(session.calls) == 1
    assert budget.used(GROQ_DEFAULT_MODEL) == 8

    # Account for prior work so only the reserved final-stage share remains.
    budget.record_usage(GROQ_DEFAULT_MODEL, {"total_tokens": 700})
    before = len(session.calls)
    with pytest.raises(GroqProviderError, match="final_script_reserve=1200"):
        client.generate_daily_story_batch("another small prompt")
    assert len(session.calls) == before

    client.generate_programme_scripts("final prompt", {"type": "object"})
    assert len(session.calls) == before + 1
