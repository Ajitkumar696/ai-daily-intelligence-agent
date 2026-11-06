import json
from datetime import date
from types import SimpleNamespace

import pytest

import src.generate_research as research_cli
import src.run_research_report as report_cli
from src.generate_research import run_research
from src.reporting.pdf_generator import (
    NoSuccessfulSummariesError,
    generate_pdf_report
)
import src.run_daily_programme as daily_pipeline
from src.research.daily_programme import (
    DAILY_PROGRAMME_SCHEMA,
    DAILY_STORY_BATCH_SCHEMA,
    MAX_DAILY_STORIES,
    _daily_context,
    _prepare_groups,
    _word_count,
    generate_daily_programme,
    validate_daily_programme
)
from src.summarization.cache import SummaryCache
from src.summarization.gemini_client import (
    GeminiClient,
    GeminiClientError
)


PROGRAMME_DATE = date(2026, 11, 2)
FIRST_URL = "https://example.com/ai-launch"
SECOND_URL = "https://news.example/ai-research"


def _narration(count, phrase):
    words = phrase.split()
    repetitions = (count + len(words) - 1) // len(words)
    return " ".join((words * repetitions)[:count])


def make_groups():
    return [
        {
            "topic": "AI model launch",
            "research_scope": "single_source",
            "sources": [{
                "publisher": "Example News",
                "title": "AI model launch",
                "url": FIRST_URL,
                "canonical_url": FIRST_URL,
                "published": PROGRAMME_DATE.isoformat(),
                "author": "Reporter",
                "content_source": "full_article",
                "content": "An AI company announced a model with new capabilities."
            }]
        },
        {
            "topic": "AI research system",
            "research_scope": "single_source",
            "sources": [{
                "publisher": "Research Daily",
                "title": "AI research system",
                "url": SECOND_URL,
                "canonical_url": SECOND_URL,
                "published": PROGRAMME_DATE.isoformat(),
                "author": "",
                "content_source": "rss_fallback",
                "content": "Researchers described a new AI research system."
            }]
        }
    ]


def test_daily_context_selects_recent_diverse_stories_with_source_urls():
    groups = []
    for index in range(60):
        publisher = "Common Publisher" if index < 50 else f"Publisher {index}"
        url = f"https://example.com/story-{index}"
        groups.append({
            "topic": f"AI development topic {index}",
            "sources": [{
                "publisher": publisher,
                "title": f"AI model development update {index}",
                "url": url,
                "canonical_url": url,
                "published": f"2026-11-{(index % 4) + 1:02d}T12:00:00+00:00",
                "content": "Useful source context. " * 100,
            }],
        })

    context = json.loads(_daily_context(_prepare_groups(groups)))

    assert len(context) == MAX_DAILY_STORIES
    urls = {source["url"] for group in context for source in group["sources"]}
    assert len(urls) == MAX_DAILY_STORIES
    assert "https://example.com/story-59" in urls
    assert any(source["publisher"] == "Publisher 59"
               for group in context for source in group["sources"])
    assert all(source["content"] for group in context for source in group["sources"])


def make_response():
    stories = []
    for story_id, topic, headline, url in (
        ("story-0001", "AI model launch", "An AI model is launched", FIRST_URL),
        ("story-0002", "AI research system", "A research system is described", SECOND_URL)
    ):
        stories.append({
            "story_id": story_id,
            "topic": topic,
            "headline": headline,
            "category": "AI development",
            "importance": "High" if story_id == "story-0001" else "Medium",
            "summary": "The provided source describes a relevant AI development.",
            "key_facts": [{
                "statement": "The source reports the described development.",
                "source_urls": [url]
            }],
            "why_it_matters": "The development may affect practical AI use.",
            "source_scope": "single_source",
            "cross_source_comparison": {
                "agreements": [],
                "contradictions": [],
                "additional_details": []
            },
            "confidence": 0.8,
            "content_quality": "One full article and one RSS fallback were supplied.",
            "source_urls": [url]
        })

    return {
        "overall_headline": "Two developments shape today's AI news",
        "executive_summary": (
            "Today's coverage includes a model announcement and an AI "
            "research development."
        ),
        "stories": stories,
        "programmes": {
            "one_minute": {
                "title": "AI News Update",
                "duration_seconds": 60,
                "narration": _narration(
                    105,
                    "First, a new model is announced, followed by a research update."
                ),
                "story_ids": ["story-0001"],
                "source_urls": [FIRST_URL]
            },
            "three_minute": {
                "title": "AI News Bulletin",
                "duration_seconds": 180,
                "narration": _narration(
                    315,
                    "The bulletin explains the model announcement and research system."
                ),
                "story_ids": ["story-0001", "story-0002"],
                "source_urls": [FIRST_URL, SECOND_URL]
            }
        }
    }


def _validation_groups(groups=None):
    return [
        {
            "story_id": f"story-{index:04d}",
            "research_scope": group["research_scope"],
            **group
        }
        for index, group in enumerate(groups or make_groups(), start=1)
    ]


class FakeDailyClient:
    model = "gemini-test"
    provider = "gemini"
    max_output_tokens = 1200

    def __init__(self, response=None, narration_responses=None):
        self.response = response or make_response()
        self.calls = []
        self.narration_responses = list(narration_responses or [])
        self.narration_calls = []

    def generate_daily_programme(self, prompt):
        self.calls.append(prompt)
        return self.response

    def generate_narration(self, prompt):
        self.narration_calls.append(prompt)
        if not self.narration_responses:
            raise AssertionError("Unexpected narration regeneration.")
        return {"narration": self.narration_responses.pop(0)}


class FakeModels:
    def __init__(self, response):
        self.responses = response if isinstance(response, list) else [response]
        self.calls = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        return self.responses.pop(0)


class FakeSDKClient:
    def __init__(self, response):
        self.models = FakeModels(response)


class FakeMessageResponse:
    def __init__(
        self,
        text,
        finish_reason=None,
        output_tokens=None,
        prompt_tokens=None,
        thoughts_tokens=None,
        total_tokens=None
    ):
        self.text = text
        self.candidates = (
            [SimpleNamespace(finish_reason=finish_reason)]
            if finish_reason is not None
            else None
        )
        self.usage_metadata = SimpleNamespace(
            candidates_token_count=output_tokens,
            prompt_token_count=prompt_tokens,
            thoughts_token_count=thoughts_tokens,
            total_token_count=total_tokens
        )


def test_daily_generation_makes_one_combined_request_and_caches(tmp_path):
    client = FakeDailyClient()
    cache_path = tmp_path / "cache.json"
    groups = make_groups()

    first = generate_daily_programme(
        groups,
        client=client,
        cache=SummaryCache(cache_path),
        programme_date=PROGRAMME_DATE
    )
    second = generate_daily_programme(
        groups,
        client=client,
        cache=SummaryCache(cache_path),
        programme_date=PROGRAMME_DATE
    )

    assert first == second
    assert len(client.calls) == 1
    assert len(first["stories"]) == 2
    assert set(first["programmes"]) == {"one_minute", "three_minute"}
    assert first["programmes"]["one_minute"]["story_ids"] == ["story-0001"]
    assert first["programmes"]["three_minute"]["story_ids"] == [
        "story-0001",
        "story-0002"
    ]
    assert first["llm_diagnostics"][0]["provider"] == "gemini"
    assert first["llm_diagnostics"][0]["model"] == "gemini-test"
    assert first["stories"][1]["content_quality_details"]["fallback_count"] == 1


def test_daily_programme_requires_valid_sources_and_both_scripts(tmp_path):
    groups = make_groups()
    response = make_response()
    response["stories"][0]["key_facts"][0]["source_urls"] = [
        "https://unrelated.example/fact"
    ]

    with pytest.raises(ValueError, match="malformed or unsupported source URL"):
        validate_daily_programme(response, _validation_groups(groups), PROGRAMME_DATE)

    response = make_response()
    response["programmes"].pop("three_minute")
    client = FakeDailyClient(response)
    cache_path = tmp_path / "invalid-cache.json"
    with pytest.raises(GeminiClientError, match="missing three_minute"):
        generate_daily_programme(
            groups,
            client=client,
            cache=SummaryCache(cache_path),
            programme_date=PROGRAMME_DATE
        )
    assert not cache_path.exists()


def test_daily_citations_from_same_story_are_validated_and_preserved():
    validated = validate_daily_programme(
        make_response(), _validation_groups(), PROGRAMME_DATE
    )

    assert validated["stories"][0]["key_facts"][0]["source_urls"] == [FIRST_URL]


def test_missing_content_quality_is_rejected_without_fabrication(tmp_path):
    response = make_response()
    response["stories"][0].pop("content_quality")

    with pytest.raises(GeminiClientError, match="content_quality must be non-empty"):
        generate_daily_programme(
        make_groups(),
        client=FakeDailyClient(response),
        cache=SummaryCache(tmp_path / "cache.json"),
        programme_date=PROGRAMME_DATE,
    )


def test_empty_content_quality_is_rejected():
    response = make_response()
    response["stories"][0]["content_quality"] = "  "

    with pytest.raises(ValueError, match="content_quality must be non-empty"):
        validate_daily_programme(response, _validation_groups(), PROGRAMME_DATE)


def test_valid_content_quality_is_preserved():
    response = make_response()
    response["stories"][0]["content_quality"] = "Full article text is available."

    validated = validate_daily_programme(
        response, _validation_groups(), PROGRAMME_DATE
    )

    assert validated["stories"][0]["content_quality"] == (
        "Full article text is available."
    )


def test_groq_batch_prompt_requires_content_quality_and_schema_keeps_field():
    from src.research.daily_programme import _daily_story_batch_prompt

    prompt = _daily_story_batch_prompt("[]", PROGRAMME_DATE)
    story_schema = DAILY_STORY_BATCH_SCHEMA["properties"]["stories"]["items"]

    assert "exactly one research object per input group" in prompt
    assert '{"stories": [' in prompt
    assert "top-level JSON value MUST be an object" in prompt
    assert "never return an array" in prompt
    assert "content_quality" in prompt
    assert "Do not output" in prompt and "headlines" in prompt
    assert "content_quality" in story_schema["properties"]
    assert "content_quality" in story_schema["required"]


def test_daily_key_fact_missing_citation_is_rejected():
    response = make_response()
    response["stories"][0]["key_facts"][0].pop("source_urls")

    with pytest.raises(ValueError, match="must contain source URL citations"):
        validate_daily_programme(response, _validation_groups(), PROGRAMME_DATE)


def test_daily_canonical_trailing_slash_citation_normalizes_to_collected_url():
    groups = make_groups()
    groups[0]["sources"][0]["canonical_url"] = FIRST_URL + "/"
    response = make_response()
    response["stories"][0]["key_facts"][0]["source_urls"] = [FIRST_URL + "/"]

    validated = validate_daily_programme(
        response, _validation_groups(groups), PROGRAMME_DATE
    )

    assert validated["stories"][0]["key_facts"][0]["source_urls"] == [FIRST_URL]
    assert validated["stories"][0]["source_urls"] == [FIRST_URL]


def test_daily_canonical_alias_cannot_borrow_another_story_url():
    groups = make_groups()
    groups[0]["sources"][0]["canonical_url"] = SECOND_URL
    response = make_response()
    response["stories"][0]["key_facts"][0]["source_urls"] = [SECOND_URL]

    with pytest.raises(ValueError, match="malformed or unsupported source URL"):
        validate_daily_programme(
            response, _validation_groups(groups), PROGRAMME_DATE
        )


def test_daily_claim_can_cite_multiple_urls_from_its_own_story():
    groups = make_groups()
    second_url = "https://example.com/ai-launch-details"
    groups[0]["sources"].append({
        **groups[0]["sources"][0],
        "title": "AI model launch details",
        "url": second_url,
        "canonical_url": second_url
    })
    response = make_response()
    response["stories"][0]["key_facts"][0]["source_urls"] = [
        FIRST_URL, second_url
    ]
    response["stories"][0]["source_urls"] = [FIRST_URL, second_url]
    response["programmes"]["one_minute"]["source_urls"] = [
        FIRST_URL, second_url
    ]
    response["programmes"]["three_minute"]["source_urls"] = [
        FIRST_URL, second_url, SECOND_URL
    ]

    validated = validate_daily_programme(
        response, _validation_groups(groups), PROGRAMME_DATE
    )

    assert validated["stories"][0]["key_facts"][0]["source_urls"] == [
        FIRST_URL, second_url
    ]


@pytest.mark.parametrize("citation", [
    "not a URL",
    "[source](https://example.com/ai-launch)",
    "https://example.com/not-collected",
    SECOND_URL
])
def test_daily_malformed_or_unsupported_citation_is_rejected(citation):
    response = make_response()
    response["stories"][0]["key_facts"][0]["source_urls"] = [citation]

    with pytest.raises(ValueError, match="malformed or unsupported source URL"):
        validate_daily_programme(response, _validation_groups(), PROGRAMME_DATE)


def test_daily_prompt_tells_model_to_omit_unverifiable_claims():
    from src.research.daily_programme import _daily_prompt

    prompt = _daily_prompt("[]", PROGRAMME_DATE)

    assert "Omit any factual claim" in prompt
    assert "omit a story if it has no verifiable key fact" in prompt
    assert "Never invent or borrow a URL from another story" in prompt


@pytest.mark.parametrize("word_count", [259, 541])
def test_three_minute_narration_outside_limits_is_regenerated(
    tmp_path,
    word_count
):
    response = make_response()
    response["programmes"]["three_minute"]["narration"] = _narration(
        word_count,
        "The original narration is outside the required word count."
    )
    replacement = _narration(
        390,
        "The publisher reports the sourced AI model development and research "
        "update, with each detail tied to the original evidence."
    )
    client = FakeDailyClient(
        response,
        narration_responses=[replacement]
    )

    programme = generate_daily_programme(
        make_groups(),
        client=client,
        cache=SummaryCache(tmp_path / "cache.json"),
        programme_date=PROGRAMME_DATE
    )

    assert len(client.narration_calls) == 1
    assert programme["programmes"]["three_minute"]["narration"] == replacement
    assert programme["programmes"]["three_minute"]["word_count"] == 390
    assert programme["programmes"]["three_minute"]["story_ids"] == [
        "story-0001",
        "story-0002"
    ]
    assert programme["programmes"]["three_minute"]["source_urls"] == [
        FIRST_URL,
        SECOND_URL
    ]
    regeneration_prompt = client.narration_calls[0]
    assert "Example News" in regeneration_prompt
    assert PROGRAMME_DATE.isoformat() in regeneration_prompt
    assert FIRST_URL in regeneration_prompt
    assert "new capabilities" in regeneration_prompt
    assert "360-420 words" in regeneration_prompt


def test_valid_three_minute_narration_is_not_regenerated(tmp_path):
    client = FakeDailyClient()
    programme = generate_daily_programme(
        make_groups(),
        client=client,
        cache=SummaryCache(tmp_path / "cache.json"),
        programme_date=PROGRAMME_DATE
    )

    assert client.narration_calls == []
    assert programme["programmes"]["three_minute"]["word_count"] == 315


def test_word_count_handles_punctuation_contractions_hyphens_and_whitespace():
    assert _word_count(
        "AI—news, don't stop; source-grounded\tupdates\nare useful."
    ) == 8


def test_second_narration_regeneration_can_recover_and_is_fully_validated(
    tmp_path,
    capsys
):
    response = make_response()
    response["programmes"]["three_minute"]["narration"] = _narration(
        200,
        "Initial narration"
    )
    recovered = _narration(
        390,
        "The original sources report specific research and model updates."
    )
    client = FakeDailyClient(
        response,
        narration_responses=[
            _narration(250, "First regeneration remains short"),
            recovered
        ]
    )

    programme = generate_daily_programme(
        make_groups(),
        client=client,
        cache=SummaryCache(tmp_path / "cache.json"),
        programme_date=PROGRAMME_DATE
    )

    assert len(client.narration_calls) == 2
    assert programme["programmes"]["three_minute"]["narration"] == recovered
    assert programme["programmes"]["three_minute"]["word_count"] == 390
    assert validate_daily_programme(
        programme,
        [
            {
                "story_id": f"story-{index:04d}",
                **group
            }
            for index, group in enumerate(make_groups(), start=1)
        ],
        PROGRAMME_DATE
    )["programmes"]["three_minute"]["word_count"] == 390
    output = capsys.readouterr().out
    assert "minimum=260" in output
    assert "maximum=540" in output
    assert "provider=gemini" in output
    assert "model=gemini-test" in output
    assert "regeneration_attempt=1/2" in output
    assert "regeneration_attempt=2/2" in output


def test_daily_gemini_request_uses_supported_schema_format():
    sdk = FakeSDKClient(FakeMessageResponse(json.dumps(make_response())))
    client = GeminiClient("gemini-test-key", sdk_client=sdk)

    assert client.generate_daily_programme("One combined daily programme.")
    request = sdk.models.calls[0]
    request_schema = request["config"]["response_json_schema"]

    assert request["config"]["response_mime_type"] == "application/json"
    assert request["config"]["max_output_tokens"] == 8192
    assert request["config"]["thinking_config"] == {
        "thinking_level": "LOW"
    }
    assert "response_schema" not in request["config"]
    assert "additional_properties" not in json.dumps(request_schema)
    assert "additionalProperties" not in json.dumps(request_schema)
    assert "one_minute" in request_schema["properties"]["programmes"]["properties"]
    assert "three_minute" in request_schema["properties"]["programmes"]["properties"]
    assert "additional_properties" not in json.dumps(DAILY_PROGRAMME_SCHEMA)


def test_daily_gemini_parses_valid_json_in_an_exact_markdown_fence():
    response_text = json.dumps(make_response())
    sdk = FakeSDKClient(
        FakeMessageResponse(f"```json\n{response_text}\n```")
    )
    client = GeminiClient("gemini-test-key", sdk_client=sdk)

    assert client.generate_daily_programme("One combined daily programme.") == (
        make_response()
    )


def test_daily_gemini_reports_sanitized_parse_details_without_retry():
    secret = "private-gemini-api-key"
    response_text = '{"partial": "DO_NOT_LEAK'
    sdk = FakeSDKClient(FakeMessageResponse(
        response_text,
        finish_reason="STOP",
        output_tokens=14,
        prompt_tokens=321,
        thoughts_tokens=87,
        total_tokens=422
    ))
    client = GeminiClient(secret, sdk_client=sdk)

    with pytest.raises(GeminiClientError) as raised:
        client.generate_daily_programme("One combined daily programme.")

    message = str(raised.value)
    assert "Gemini returned invalid JSON" in message
    assert "parse_error=" in message
    assert "line=1" in message
    assert "column=" in message
    assert "response_chars=" in message
    assert "finish_reason=STOP" in message
    assert "output_tokens=14" in message
    assert "prompt_tokens=321" in message
    assert "thoughts_tokens=87" in message
    assert "total_tokens=422" in message
    assert "DO_NOT_LEAK" not in message
    assert secret not in message
    assert len(sdk.models.calls) == 1


def test_daily_gemini_retries_once_only_for_explicit_token_truncation():
    from google.genai.types import FinishReason

    sdk = FakeSDKClient([
        FakeMessageResponse(
            '{"partial":',
            finish_reason=FinishReason.MAX_TOKENS,
            output_tokens=8192
        ),
        FakeMessageResponse(
            json.dumps(make_response()),
            finish_reason=FinishReason.STOP
        )
    ])
    client = GeminiClient("gemini-test-key", sdk_client=sdk)

    result = client.generate_daily_programme(
        "One combined daily programme."
    )

    assert result == make_response()
    assert len(sdk.models.calls) == 2
    assert all(
        call["config"]["max_output_tokens"] == expected
        for call, expected in zip(sdk.models.calls, (8192, 16384))
    )
    assert "Regenerate exactly one complete JSON object" in (
        sdk.models.calls[1]["contents"]
    )
    assert "MAX_TOKENS" not in sdk.models.calls[1]["contents"]
    assert "response_json_schema" not in sdk.models.calls[1]["contents"]
    assert all(
        call["config"]["response_mime_type"] == "application/json"
        and call["config"]["response_json_schema"] == DAILY_PROGRAMME_SCHEMA
        and call["config"]["thinking_config"]["thinking_level"] == "LOW"
        for call in sdk.models.calls
    )


def test_daily_gemini_does_not_retry_second_truncated_response():
    from google.genai.types import FinishReason

    sdk = FakeSDKClient([
        FakeMessageResponse(
            '{"partial":',
            finish_reason=FinishReason.MAX_TOKENS
        ),
        FakeMessageResponse(
            '{"still_partial":',
            finish_reason=FinishReason.MAX_TOKENS
        )
    ])
    client = GeminiClient("gemini-test-key", sdk_client=sdk)

    with pytest.raises(GeminiClientError, match="attempt=2"):
        client.generate_daily_programme(
            "One combined daily programme."
        )

    assert len(sdk.models.calls) == 2
    assert [
        call["config"]["max_output_tokens"]
        for call in sdk.models.calls
    ] == [8192, 16384]


def test_daily_gemini_recovery_increases_configured_token_budget():
    from google.genai.types import FinishReason

    sdk = FakeSDKClient([
        FakeMessageResponse(
            '{"partial":',
            finish_reason=FinishReason.MAX_TOKENS,
            output_tokens=29,
            prompt_tokens=900,
            thoughts_tokens=1160,
            total_tokens=2089
        ),
        FakeMessageResponse(
            '{"still_partial":',
            finish_reason=FinishReason.MAX_TOKENS,
            output_tokens=29,
            prompt_tokens=900,
            thoughts_tokens=1160,
            total_tokens=2089
        )
    ])
    client = GeminiClient(
        "gemini-test-key",
        max_output_tokens=1200,
        sdk_client=sdk
    )

    with pytest.raises(GeminiClientError) as raised:
        client.generate_daily_programme("One combined daily programme.")

    assert [
        call["config"]["max_output_tokens"]
        for call in sdk.models.calls
    ] == [1200, 8192]
    assert "finish_reason=MAX_TOKENS" in str(raised.value)
    assert "output_tokens=29" in str(raised.value)
    assert "prompt_tokens=900" in str(raised.value)
    assert "thoughts_tokens=1160" in str(raised.value)
    assert "total_tokens=2089" in str(raised.value)
    assert "max_output_tokens=8192" in str(raised.value)


def test_daily_report_is_one_pdf_with_both_scripts_and_clickable_sources(
    tmp_path
):
    client = FakeDailyClient()
    programme = generate_daily_programme(
        make_groups(),
        client=client,
        cache=SummaryCache(tmp_path / "cache.json"),
        programme_date=PROGRAMME_DATE
    )
    pdf_path = generate_pdf_report(
        programme,
        output_dir=tmp_path / "reports",
        report_date=PROGRAMME_DATE
    )

    pdf_content = pdf_path.read_bytes()
    assert pdf_content.startswith(b"%PDF")
    assert b"/URI" in pdf_content
    assert FIRST_URL.encode() in pdf_content
    assert SECOND_URL.encode() in pdf_content
    assert len(list((tmp_path / "reports").glob("*.pdf"))) == 1


def test_empty_daily_package_never_creates_a_pdf(tmp_path):
    programme = make_response()
    programme["stories"] = []
    client = FakeDailyClient(programme)

    with pytest.raises(GeminiClientError, match="No successfully researched"):
        generate_daily_programme(
            make_groups(),
            client=client,
            cache=SummaryCache(tmp_path / "cache.json"),
            programme_date=PROGRAMME_DATE
        )

    with pytest.raises(NoSuccessfulSummariesError):
        generate_pdf_report(
            [],
            output_dir=tmp_path / "reports",
            report_date=PROGRAMME_DATE
        )
    assert not (tmp_path / "reports").exists()


def test_daily_research_pipeline_saves_combined_json(tmp_path):
    input_file = tmp_path / "updates.json"
    output_file = tmp_path / "daily_programme.json"
    input_file.write_text(json.dumps([
        {
            "source": "Example News",
            "publisher": "Example News",
            "source_type": "rss",
            "title": "AI model launch",
            "url": FIRST_URL,
            "published": PROGRAMME_DATE.isoformat(),
            "summary": "An AI company announced a model with new capabilities."
        },
        {
            "source": "Research Daily",
            "publisher": "Research Daily",
            "source_type": "rss",
            "title": "AI research system",
            "url": SECOND_URL,
            "published": PROGRAMME_DATE.isoformat(),
            "summary": "Researchers described a new AI research system."
        }
    ]), encoding="utf-8")
    client = FakeDailyClient()

    groups, programme, saved_path = run_research(
        input_file=input_file,
        output_file=output_file,
        client=client,
        cache=SummaryCache(tmp_path / "cache.json"),
        programme_date=PROGRAMME_DATE
    )

    assert len(groups) == 2
    assert len(client.calls) == 1
    assert saved_path == output_file
    assert json.loads(output_file.read_text(encoding="utf-8")) == programme
    assert programme["programme_status"] == "SUCCESS"


def _many_research_groups(count):
    groups = []
    for index in range(count):
        url = f"https://example.com/topic-{index}"
        groups.append({
            "topic": f"AI topic {index}",
            "sources": [{
                "publisher": f"Publisher {index}",
                "title": f"AI update {index}",
                "url": url,
                "canonical_url": url,
                "published": PROGRAMME_DATE.isoformat(),
                "content_source": "full_article",
                "content": f"Publisher reports a supported AI development {index}."
            }]
        })
    return groups


class FakeBatchedGroqClient:
    provider = "groq"
    model = "openai/gpt-oss-20b"

    def __init__(self, fail_batch=None, topic_mode="valid"):
        self.fail_batch = fail_batch
        self.topic_mode = topic_mode
        self.story_prompts = []
        self.script_prompts = []
        self.api_call_count = 0
        self.last_usage = {}

    def generate_daily_story_batch(self, prompt):
        from src.summarization.providers import GroqProviderError
        self.story_prompts.append(prompt)
        self.api_call_count += 1
        if self.fail_batch == len(self.story_prompts):
            raise GroqProviderError(
                "Groq API error HTTP 400. API detail: invalid schema"
            )
        packet = json.loads(prompt.split("Grouped collected coverage:\n", 1)[1])
        self.last_usage = {"completion_tokens": 200}
        stories = []
        for group in packet:
            story = {
                "category": "AI News",
                "importance": "Medium",
                "summary": f"The source reports a supported development for {group['topic']}.",
                "key_facts": [{
                    "statement": f"The publisher reported development {group['story_id']}.",
                    "source_indices": [0],
                }],
                "why_it_matters": "The reported development may matter to AI users.",
                "cross_source_comparison": {
                    "agreements": [],
                    "contradictions": [],
                    "additional_details": [],
                },
                "confidence": 0.8,
                "content_quality": "A source article was supplied.",
            }
            stories.append(story)
        return {"stories": stories}

    def generate_programme_scripts(self, prompt, schema):
        self.script_prompts.append((prompt, schema))
        self.api_call_count += 1
        if "segments" in schema.get("properties", {}):
            packet = json.loads(
                prompt.split("Validated story evidence:\n", 1)[1]
            )
            return {"segments": [{
                "story_id": item["story_id"],
                "one_minute_segment": _narration(
                    item["one_minute_target_words"],
                    f"One minute supported report {item['story_id']}."
                ) if item["one_minute_needed"] else "",
                "three_minute_segment": _narration(
                    item["three_minute_target_words"],
                    f"Three minute supported report {item['story_id']}."
                ),
            } for item in packet]}
        packet = json.loads(prompt.split("Validated story evidence:\n", 1)[1])
        story_ids = [story["story_id"] for story in packet]
        urls_by_id = {
            story["story_id"]: story["source_urls"] for story in packet
        }
        one_ids = story_ids[:min(3, len(story_ids))]
        one_urls = list(dict.fromkeys(
            url for story_id in one_ids for url in urls_by_id[story_id]
        ))
        all_urls = list(dict.fromkeys(
            url for story_id in story_ids for url in urls_by_id[story_id]
        ))
        self.last_usage = {"completion_tokens": 300}
        return {
            "overall_headline": "Validated AI developments",
            "executive_summary": "The selected sources report these AI developments.",
            "programmes": {
                "one_minute": {
                    "title": "AI news update",
                    "duration_seconds": 60,
                    "narration": _narration(100, "The sources report verified AI developments."),
                    "story_ids": one_ids,
                    "source_urls": one_urls,
                },
                "three_minute": {
                    "title": "AI news bulletin",
                    "duration_seconds": 180,
                    "narration": _narration(300, "The sources report verified AI developments with useful context."),
                    "story_ids": story_ids,
                    "source_urls": all_urls,
                },
            },
        }


def test_groq_research_batches_validate_and_combine_all_selected_topics(tmp_path):
    from src.research.daily_programme import validate_daily_programme_record

    client = FakeBatchedGroqClient()
    programme = generate_daily_programme(
        _many_research_groups(4),
        client=client,
        cache=SummaryCache(tmp_path / "cache.json"),
        programme_date=PROGRAMME_DATE,
    )
    validated = validate_daily_programme_record(programme)

    assert len(client.story_prompts) == 2
    assert all(len(prompt) <= 22000 for prompt in client.story_prompts)
    assert [
        len(json.loads(prompt.split("Grouped collected coverage:\n", 1)[1]))
        for prompt in client.story_prompts
    ] == [3, 1]
    assert len(client.script_prompts) == 1
    assert len(validated["stories"]) == 4
    assert len(programme["llm_diagnostics"]) == 3
    assert all(item["provider"] == "groq" for item in programme["llm_diagnostics"])


@pytest.mark.parametrize("topic_mode", ["missing", "empty", "whitespace"])
def test_groq_batch_restores_missing_or_blank_topic_from_group_metadata(
    tmp_path, topic_mode
):
    groups = _many_research_groups(1)
    programme = generate_daily_programme(
        groups,
        client=FakeBatchedGroqClient(topic_mode=topic_mode),
        cache=SummaryCache(tmp_path / f"cache-{topic_mode}.json"),
        programme_date=PROGRAMME_DATE,
    )

    assert programme["stories"][0]["topic"] == groups[0]["topic"]


def test_groq_batch_does_not_keep_model_topic_when_trusted_topic_exists(tmp_path):
    groups = _many_research_groups(1)
    programme = generate_daily_programme(
        groups,
        client=FakeBatchedGroqClient(topic_mode="untrusted"),
        cache=SummaryCache(tmp_path / "cache-untrusted.json"),
        programme_date=PROGRAMME_DATE,
    )

    assert programme["stories"][0]["topic"] == groups[0]["topic"]


@pytest.mark.parametrize("value_mode", ["missing", "empty", "whitespace"])
def test_batch_rejects_missing_or_blank_generated_research_fields(
    tmp_path, value_mode
):
    class InvalidContentBatch(FakeBatchedGroqClient):
        def generate_daily_story_batch(self, prompt):
            result = super().generate_daily_story_batch(prompt)
            story = result["stories"][0]
            if value_mode == "missing":
                story.pop("summary")
            elif value_mode == "empty":
                story["summary"] = ""
            else:
                story["summary"] = "   "
            return result

    with pytest.raises(GeminiClientError, match="story_batch batch=1/1") as caught:
        generate_daily_programme(
            _many_research_groups(1), client=InvalidContentBatch(),
            cache=SummaryCache(tmp_path / f"invalid-{value_mode}.json"),
            programme_date=PROGRAMME_DATE
        )
    assert "summary" in str(caught.value)
    assert "story_id=story-0001" in str(caught.value)


@pytest.mark.parametrize("value_mode", ["missing", "empty", "whitespace"])
def test_daily_cache_rejects_missing_or_blank_research_content(
    tmp_path, value_mode
):
    from src.research.daily_programme import _daily_context, _prepare_groups
    from src.summarization.cache import content_fingerprint

    groups = _many_research_groups(1)
    cache = SummaryCache(tmp_path / "cache.json")
    generated = generate_daily_programme(
        groups, client=FakeBatchedGroqClient(), cache=cache,
        programme_date=PROGRAMME_DATE
    )
    fingerprint = content_fingerprint(
        PROGRAMME_DATE.isoformat(), _daily_context(_prepare_groups(groups))
    )
    stale = dict(generated)
    stale["stories"] = [dict(generated["stories"][0])]
    if value_mode == "missing":
        stale["stories"][0].pop("content_quality")
    elif value_mode == "empty":
        stale["stories"][0]["content_quality"] = ""
    else:
        stale["stories"][0]["content_quality"] = "   "
    cache.set(
        f"daily-programme:v2:{PROGRAMME_DATE.isoformat()}",
        fingerprint,
        stale
    )

    with pytest.raises(ValueError, match="cache_validation failed.*content_quality"):
        generate_daily_programme(
            groups, cache=cache, programme_date=PROGRAMME_DATE
        )


def test_daily_cache_restores_trusted_story_metadata(tmp_path):
    from src.research.daily_programme import _daily_context, _prepare_groups
    from src.summarization.cache import content_fingerprint

    groups = _many_research_groups(1)
    cache = SummaryCache(tmp_path / "cache.json")
    generated = generate_daily_programme(
        groups, client=FakeBatchedGroqClient(), cache=cache,
        programme_date=PROGRAMME_DATE
    )
    fingerprint = content_fingerprint(
        PROGRAMME_DATE.isoformat(),
        _daily_context(_prepare_groups(groups))
    )
    key = f"daily-programme:v2:{PROGRAMME_DATE.isoformat()}"
    stale = dict(generated)
    stale["stories"] = [dict(generated["stories"][0])]
    for field in ("headline", "topic", "source_scope", "source_urls"):
        stale["stories"][0].pop(field, None)
    cache.set(key, fingerprint, stale)

    cached = generate_daily_programme(
        groups, cache=cache, programme_date=PROGRAMME_DATE
    )

    assert cached["stories"][0]["headline"] == "AI update 0"
    assert cached["stories"][0]["topic"] == "AI topic 0"
    assert cached["stories"][0]["source_urls"] == [
        "https://example.com/topic-0"
    ]


def test_missing_topic_without_trusted_metadata_is_rejected_without_inference():
    from src.research.daily_programme import _prepare_groups

    groups = make_groups()
    groups[0]["topic"] = "  "
    groups[0]["headline"] = ""
    groups[0]["sources"][0]["title"] = ""
    with pytest.raises(
        ValueError, match="has no trusted topic metadata"
    ):
        _prepare_groups(groups)


def test_batch_validation_error_names_stage_and_batch_without_source_text(tmp_path):
    class InvalidBatchClient(FakeBatchedGroqClient):
        def generate_daily_story_batch(self, prompt):
            response = super().generate_daily_story_batch(prompt)
            response["stories"][0]["headline"] = "model supplied metadata"
            return response

    with pytest.raises(GeminiClientError) as caught:
        generate_daily_programme(
            _many_research_groups(1),
            client=InvalidBatchClient(),
            cache=SummaryCache(tmp_path / "cache.json"),
            programme_date=PROGRAMME_DATE,
        )

    message = str(caught.value)
    assert "story_batch batch=1/1 story_ids=story-0001 validation failed" in message
    assert "non-content fields" in message
    assert "Publisher reports a supported AI development" not in message


def test_research_batch_failure_preserves_existing_programme(tmp_path):
    from src.summarization.providers import GroqProviderError

    input_file = tmp_path / "updates.json"
    output_file = tmp_path / "daily_programme.json"
    input_articles = []
    for group in _many_research_groups(4):
        source = group["sources"][0]
        input_articles.append({
            "source": source["publisher"],
            "publisher": source["publisher"],
            "title": source["title"],
            "url": source["url"],
            "published": source["published"],
            "summary": source["content"],
        })
    input_file.write_text(json.dumps(input_articles), encoding="utf-8")
    original = b'{"existing":"programme"}\n'
    output_file.write_bytes(original)
    client = FakeBatchedGroqClient(fail_batch=2)

    with pytest.raises(GroqProviderError, match="HTTP 400"):
        run_research(
            input_file=input_file,
            output_file=output_file,
            client=client,
            cache=SummaryCache(tmp_path / "cache.json"),
            programme_date=PROGRAMME_DATE,
        )

    assert output_file.read_bytes() == original


def test_failed_daily_research_never_saves_programme_or_pdf(tmp_path):
    from src.run_research_report import run_research_report

    input_file = tmp_path / "updates.json"
    output_file = tmp_path / "daily_programme.json"
    report_dir = tmp_path / "reports"
    input_file.write_text(json.dumps([
        {
            "source": "Example News",
            "publisher": "Example News",
            "source_type": "rss",
            "title": "AI model launch",
            "url": FIRST_URL,
            "published": PROGRAMME_DATE.isoformat(),
            "summary": "An AI company announced a model with new capabilities."
        }
    ]), encoding="utf-8")
    previous_programme = make_response()
    output_file.write_text(
        json.dumps(previous_programme),
        encoding="utf-8"
    )

    invalid_response = make_response()
    invalid_response["stories"] = [
        story
        for story in invalid_response["stories"]
        if story["story_id"] == "story-0001"
    ]
    invalid_response["programmes"]["three_minute"]["narration"] = _narration(
        200,
        "Too short"
    )
    invalid_response["programmes"]["three_minute"]["story_ids"] = [
        "story-0001"
    ]
    invalid_response["programmes"]["three_minute"]["source_urls"] = [FIRST_URL]

    class FailedClient:
        model = "gemini-failing-test"
        provider = "gemini"

        def generate_daily_programme(self, prompt):
            return invalid_response

        def generate_narration(self, prompt):
            return {"narration": _narration(259, "Still too short")}

    cache = SummaryCache(tmp_path / "cache.json")
    with pytest.raises(
        GeminiClientError,
        match=r"word_count=259.*minimum=260.*maximum=540.*"
        r"provider=gemini, model=gemini-failing-test"
    ):
        run_research_report(
            input_file=input_file,
            brief_file=output_file,
            report_dir=report_dir,
            client=FailedClient(),
            cache=cache,
            max_topics=1
        )

    assert json.loads(output_file.read_text(encoding="utf-8")) == previous_programme
    assert not cache.path.exists()
    assert not report_dir.exists()


def test_exhausted_mocked_gemini_retries_preserve_existing_daily_programme(
    tmp_path,
    monkeypatch
):
    from google.genai.errors import ServerError
    import src.summarization.gemini_client as gemini_client_module

    input_file = tmp_path / "updates.json"
    output_file = tmp_path / "daily_programme.json"
    cache_path = tmp_path / "cache.json"
    existing = make_response()
    output_file.write_text(json.dumps(existing), encoding="utf-8")
    input_file.write_text(json.dumps([{
        "source": "Example News",
        "publisher": "Example News",
        "source_type": "rss",
        "title": "AI model launch",
        "url": FIRST_URL,
        "published": PROGRAMME_DATE.isoformat(),
        "summary": "An AI company announced a model with new capabilities."
    }]), encoding="utf-8")
    monkeypatch.setattr(gemini_client_module.time, "sleep", lambda delay: None)
    class FailedModels:
        def __init__(self):
            self.calls = []

        def generate_content(self, **kwargs):
            self.calls.append(kwargs)
            raise ServerError(503, {"error": {
                "status": "UNAVAILABLE",
                "message": "Temporary service failure"
            }})

    sdk = SimpleNamespace(models=FailedModels())
    client = GeminiClient("gemini-test-key", sdk_client=sdk)

    with pytest.raises(GeminiClientError, match="gemini-3.8-flash"):
        run_research(
            input_file=input_file,
            output_file=output_file,
            client=client,
            cache=SummaryCache(cache_path),
            programme_date=PROGRAMME_DATE
        )

    assert json.loads(output_file.read_text(encoding="utf-8")) == existing
    assert not cache_path.exists()
    assert len(sdk.models.calls) == 5


def test_end_to_end_runner_collects_then_creates_one_daily_package(
    tmp_path,
    monkeypatch
):
    collection = type(
        "Collection",
        (),
        {"saved_articles": [{"title": "One collected story"}]}
    )()
    calls = []

    def fake_collection(**kwargs):
        calls.append(("collect", kwargs))
        return collection

    def fake_research(**kwargs):
        calls.append(("research", kwargs))
        return (
            {"stories": [{}]},
            tmp_path / "daily_programme.json",
            tmp_path / "daily.pdf"
        )

    monkeypatch.setattr(daily_pipeline, "run_pipeline", fake_collection)
    monkeypatch.setattr(
        daily_pipeline,
        "run_research_report",
        fake_research
    )
    result = daily_pipeline.run_daily_programme(
        input_file=tmp_path / "updates.json",
        programme_file=tmp_path / "daily_programme.json",
        report_dir=tmp_path / "reports",
        history_file=tmp_path / "history.json",
        max_topics=1
    )

    assert result[2] == tmp_path / "daily.pdf"
    assert [call[0] for call in calls] == ["collect", "research"]
    assert calls[0][1]["output_file"] == tmp_path / "updates.json"
    assert calls[1][1]["max_topics"] == 1


def test_end_to_end_runner_stops_when_collection_is_empty(monkeypatch):
    monkeypatch.setattr(
        daily_pipeline,
        "run_pipeline",
        lambda **kwargs: type("Collection", (), {"saved_articles": []})()
    )
    monkeypatch.setattr(
        daily_pipeline,
        "run_research_report",
        lambda **kwargs: pytest.fail("research should not run without articles")
    )

    with pytest.raises(ValueError, match="No new articles"):
        daily_pipeline.run_daily_programme()


@pytest.mark.parametrize("cli_module", [research_cli, report_cli])
def test_research_clis_exit_cleanly_without_outputs_on_gemini_failure(
    cli_module,
    tmp_path,
    monkeypatch,
    capsys
):
    output_file = tmp_path / "daily_programme.json"
    pdf_dir = tmp_path / "reports"

    def fail_research(*args, **kwargs):
        raise GeminiClientError("Gemini API error HTTP 503 (UNAVAILABLE)")

    monkeypatch.setattr(cli_module, "run_research", fail_research)
    if cli_module is report_cli:
        monkeypatch.setattr(
            cli_module,
            "generate_pdf_report",
            lambda *args, **kwargs: pytest.fail(
                "The PDF must not be generated when research fails."
            )
        )
        args = [
            "--brief-output", str(output_file),
            "--report-dir", str(pdf_dir)
        ]
    else:
        args = ["--output", str(output_file)]

    with pytest.raises(SystemExit) as caught:
        cli_module.main(args)

    captured = capsys.readouterr()
    assert caught.value.code == 2
    assert "AI research failed" in captured.err
    assert "HTTP 503" in captured.err
    assert "Traceback" not in captured.err
    assert not output_file.exists()
    assert not pdf_dir.exists()


# --- batch recovery: empty key_facts, retry, split, skip -------------------

class _EmptyFactsClient(FakeBatchedGroqClient):
    """Returns empty key_facts for chosen story ids on the first `bad_calls`
    calls that include them; `always_bad` ids are bad on every call."""

    def __init__(self, bad_first_calls=0, always_bad=()):
        super().__init__()
        self.bad_first_calls = bad_first_calls
        self.always_bad = set(always_bad)

    def generate_daily_story_batch(self, prompt):
        result = super().generate_daily_story_batch(prompt)
        packet = json.loads(prompt.split("Grouped collected coverage:\n", 1)[1])
        for story, group in zip(result["stories"], packet):
            if group["story_id"] in self.always_bad or (
                len(self.story_prompts) <= self.bad_first_calls
            ):
                story["key_facts"] = []
        return result


def test_empty_key_facts_batch_is_retried_and_recovers(tmp_path):
    client = _EmptyFactsClient(bad_first_calls=1)
    programme = generate_daily_programme(
        _many_research_groups(3),
        client=client,
        cache=SummaryCache(tmp_path / "cache.json"),
        programme_date=PROGRAMME_DATE,
    )

    assert len(programme["stories"]) == 3
    assert len(client.story_prompts) == 2          # first rejected, retry ok
    assert not any(
        d.get("stage") == "story_batch_skipped"
        for d in programme["llm_diagnostics"]
    )


def test_one_bad_story_is_skipped_without_losing_the_rest(tmp_path):
    client = _EmptyFactsClient(always_bad={"story-0002"})
    programme = generate_daily_programme(
        _many_research_groups(3),
        client=client,
        cache=SummaryCache(tmp_path / "cache.json"),
        programme_date=PROGRAMME_DATE,
    )

    assert [s["story_id"] for s in programme["stories"]] == [
        "story-0001", "story-0003",
    ]
    skipped = [
        d for d in programme["llm_diagnostics"]
        if d.get("stage") == "story_batch_skipped"
    ]
    assert len(skipped) == 1
    assert skipped[0]["story_ids"] == ["story-0002"]
    assert "source-cited key facts" in skipped[0]["error"]


def test_all_stories_failing_still_raises_labelled_error(tmp_path):
    client = _EmptyFactsClient(
        always_bad={"story-0001", "story-0002"}
    )
    with pytest.raises(GeminiClientError) as caught:
        generate_daily_programme(
            _many_research_groups(2),
            client=client,
            cache=SummaryCache(tmp_path / "cache.json"),
            programme_date=PROGRAMME_DATE,
        )

    message = str(caught.value)
    assert "story_batch batch=1/1" in message
    assert "source-cited key facts" in message


def test_provider_errors_are_not_retried_as_content_failures(tmp_path):
    client = FakeBatchedGroqClient(fail_batch=1)
    from src.summarization.providers import GroqProviderError
    with pytest.raises(GroqProviderError):
        generate_daily_programme(
            _many_research_groups(3),
            client=client,
            cache=SummaryCache(tmp_path / "cache.json"),
            programme_date=PROGRAMME_DATE,
        )
    assert len(client.story_prompts) == 1


def test_interrupted_batch_run_resumes_from_validated_checkpoint(tmp_path):
    from src.summarization.providers import GroqProviderError

    cache = SummaryCache(tmp_path / "resume-cache.json")
    interrupted = FakeBatchedGroqClient(fail_batch=2)
    groups = _many_research_groups(4)
    with pytest.raises(GroqProviderError, match="HTTP 400"):
        generate_daily_programme(
            groups,
            client=interrupted,
            cache=cache,
            programme_date=PROGRAMME_DATE,
        )
    assert len(interrupted.story_prompts) == 2

    resumed = FakeBatchedGroqClient()
    programme = generate_daily_programme(
        groups,
        client=resumed,
        cache=SummaryCache(tmp_path / "resume-cache.json"),
        programme_date=PROGRAMME_DATE,
    )

    assert len(programme["stories"]) == 4
    assert len(resumed.story_prompts) == 1
    resumed_packet = json.loads(
        resumed.story_prompts[0].split("Grouped collected coverage:\n", 1)[1]
    )
    assert [item["story_id"] for item in resumed_packet] == ["story-0004"]


def test_story_batch_prompt_requires_non_empty_key_facts():
    from src.research.daily_programme import _daily_story_batch_prompt

    prompt = _daily_story_batch_prompt("[]", PROGRAMME_DATE)
    assert "never return an empty key_facts array" in prompt


# --- pre-filter: groups with no usable source text -------------------------

def _metadata_only_group(index=99):
    return {
        "topic": f"No text topic {index}",
        "sources": [{
            "publisher": "Metadata Publisher",
            "title": f"Metadata only {index}",
            "url": f"https://example.com/meta-{index}",
            "published": PROGRAMME_DATE.isoformat(),
            "content_source": "metadata_only",
            "content": "",
        }],
    }


def test_metadata_only_group_is_never_sent_to_the_model(tmp_path):
    client = FakeBatchedGroqClient()
    groups = _many_research_groups(2) + [_metadata_only_group()]
    programme = generate_daily_programme(
        groups,
        client=client,
        cache=SummaryCache(tmp_path / "cache.json"),
        programme_date=PROGRAMME_DATE,
    )

    sent_ids = [
        item["story_id"]
        for prompt in client.story_prompts
        for item in json.loads(prompt.split("Grouped collected coverage:\n", 1)[1])
    ]
    assert len(programme["stories"]) == 2
    assert len(sent_ids) == 2
    skipped = [
        d for d in programme["llm_diagnostics"]
        if d.get("stage") == "pre_filter_no_source_text"
    ]
    assert len(skipped) == 1
    assert skipped[0]["reason"] == "insufficient_source_text"
    assert len(skipped[0]["story_ids"]) == 1


def test_only_metadata_only_groups_raises_clear_error(tmp_path):
    with pytest.raises(ValueError, match="usable source text"):
        generate_daily_programme(
            [_metadata_only_group(1), _metadata_only_group(2)],
            client=FakeBatchedGroqClient(),
            cache=SummaryCache(tmp_path / "cache.json"),
            programme_date=PROGRAMME_DATE,
        )


# --- script-stage input budget ---------------------------------------------

def _validated_story(index, text_len=900, urls=4):
    url_list = [
        f"https://example.com/story-{index}/source-{n}/" + "x" * 80
        for n in range(urls)
    ]
    return {
        "story_id": f"story-{index:04d}",
        "headline": f"Headline {index} " + "h" * 60,
        "summary": "s" * text_len,
        "why_it_matters": "w" * text_len,
        "key_facts": [
            {"statement": "f" * text_len, "source_urls": url_list}
            for _ in range(3)
        ],
        "source_urls": url_list,
    }


def test_script_prompt_omits_per_fact_urls_but_keeps_story_urls():
    from src.research.daily_programme import _programme_script_prompt

    story = _validated_story(1, text_len=50, urls=2)
    prompt = _programme_script_prompt([story], PROGRAMME_DATE)
    packet = json.loads(prompt.split("Validated story evidence:\n", 1)[1])

    assert packet[0]["source_urls"] == story["source_urls"]
    assert all(set(f) == {"statement"} for f in packet[0]["key_facts"])


def test_large_story_set_is_partitioned_without_omissions():
    from src.research.daily_programme import (
        MAX_SCRIPT_PROMPT_CHARS,
        _programme_script_chunks,
        _script_segments_prompt,
    )

    stories = [_validated_story(i) for i in range(30)]
    minute_ids = {story["story_id"] for story in stories[:3]}
    minute_target = 34
    three_target = 12
    chunks = _programme_script_chunks(
        stories, PROGRAMME_DATE, minute_ids, minute_target, three_target
    )

    assert [story for chunk in chunks for story in chunk] == stories
    assert all(len(chunk) < len(stories) for chunk in chunks)
    assert all(
        len(_script_segments_prompt(
            chunk, PROGRAMME_DATE, minute_ids, minute_target, three_target
        )) <= MAX_SCRIPT_PROMPT_CHARS
        for chunk in chunks
    )


def test_single_unbounded_story_fails_clearly_without_dropping_it():
    from src.research.daily_programme import _programme_script_chunks

    huge = _validated_story(1, text_len=12000, urls=1)
    with pytest.raises(ValueError, match="story_id=story-0001.*no stories were omitted"):
        _programme_script_chunks(
            [huge], PROGRAMME_DATE, {huge["story_id"]}, 100, 340
        )


def test_end_to_end_long_research_keeps_every_scripted_story(tmp_path):
    class LongResearchClient(FakeBatchedGroqClient):
        def generate_daily_story_batch(self, prompt):
            result = super().generate_daily_story_batch(prompt)
            for story in result["stories"]:
                story["summary"] = "s" * 900
                story["why_it_matters"] = "w" * 900
                story["key_facts"] = [
                    {"statement": "f" * 900, "source_indices": [0]}
                    for _ in range(3)
                ]
            return result

    client = LongResearchClient()
    programme = generate_daily_programme(
        _many_research_groups(30),
        client=client,
        cache=SummaryCache(tmp_path / "cache.json"),
        programme_date=PROGRAMME_DATE,
    )

    assert len(programme["stories"]) == MAX_DAILY_STORIES
    all_story_ids = [story["story_id"] for story in programme["stories"]]
    assert programme["programmes"]["three_minute"]["story_ids"] == all_story_ids
    all_urls = list(dict.fromkeys(
        url for story in programme["stories"] for url in story["source_urls"]
    ))
    assert programme["programmes"]["three_minute"]["source_urls"] == all_urls
    for story in programme["stories"]:
        assert story["headline"].startswith("AI update")
        assert story["key_facts"]
    trimmed = [
        d for d in programme["llm_diagnostics"]
        if d.get("stage") == "script_budget_trimmed"
    ]
    assert trimmed == []


def test_final_script_budget_exhaustion_resumes_cached_chunks_without_omissions(
    tmp_path
):
    from src.summarization.providers import GroqProviderError

    class LongResearchClient(FakeBatchedGroqClient):
        def __init__(self, fail_second_segment=False):
            super().__init__()
            self.fail_second_segment = fail_second_segment
            self.segment_story_ids = []
            self.completed_segment_story_ids = []

        def generate_daily_story_batch(self, prompt):
            result = super().generate_daily_story_batch(prompt)
            for story in result["stories"]:
                story["summary"] = "s" * 1200
                story["why_it_matters"] = "w" * 1200
                story["key_facts"] = [{
                    "statement": "f" * 1200,
                    "source_indices": [0],
                } for _ in range(3)]
            return result

        def generate_programme_scripts(self, prompt, schema):
            if "segments" in schema.get("properties", {}):
                packet = json.loads(
                    prompt.split("Validated story evidence:\n", 1)[1]
                )
                self.segment_story_ids.extend(
                    item["story_id"] for item in packet
                )
                if self.fail_second_segment and len(self.script_prompts) == 1:
                    raise GroqProviderError(
                        "budget exhausted", error_type="TOKEN_BUDGET_EXHAUSTED",
                        stage="final_script",
                    )
                result = super().generate_programme_scripts(prompt, schema)
                self.completed_segment_story_ids.extend(
                    item["story_id"] for item in packet
                )
                return result
            return super().generate_programme_scripts(prompt, schema)

    cache_path = tmp_path / "resume-final-script.json"
    groups = _many_research_groups(30)
    interrupted = LongResearchClient(fail_second_segment=True)
    with pytest.raises(GroqProviderError, match="remaining story_ids"):
        generate_daily_programme(
            groups,
            client=interrupted,
            cache=SummaryCache(cache_path),
            programme_date=PROGRAMME_DATE,
        )
    completed_script_ids = set(interrupted.completed_segment_story_ids)
    assert completed_script_ids

    resumed = LongResearchClient()
    programme = generate_daily_programme(
        groups,
        client=resumed,
        cache=SummaryCache(cache_path),
        programme_date=PROGRAMME_DATE,
    )

    all_ids = [story["story_id"] for story in programme["stories"]]
    assert resumed.story_prompts == []
    assert not (completed_script_ids & set(resumed.segment_story_ids))
    assert programme["programmes"]["three_minute"]["story_ids"] == all_ids
    assert set(
        interrupted.completed_segment_story_ids + resumed.segment_story_ids
    ) == set(all_ids)
