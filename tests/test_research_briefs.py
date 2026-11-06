import json

import pytest

from src.research.briefs import generate_research_briefs
from src.research.schemas import (
    RESEARCH_OUTPUT_JSON_SCHEMA,
    validate_research_output
)
from src.summarization.cache import SummaryCache
from src.summarization.gemini_client import GeminiClient


SOURCE_URL = "https://example.com/ai-launch"
SECOND_SOURCE_URL = "https://news.example/ai-launch"


def sourced_claim(statement, urls=(SOURCE_URL,)):
    return {"statement": statement, "source_urls": list(urls)}


def generated_output(urls=(SOURCE_URL,)):
    claim = sourced_claim("The source reports a documented announcement.", urls)
    brief = {
        "topic": "AI model launch",
        "headline": "A new model is announced",
        "background": [claim],
        "what_happened": [claim],
        "technology_explanation": [claim],
        "important_developments": [
            sourced_claim(f"Development {index} is reported.", urls)
            for index in range(1, 4)
        ],
        "technical_details": [claim],
        "companies_models_technologies": [claim],
        "real_world_applications": [claim],
        "benefits": [claim],
        "limitations": [claim],
        "potential_impact": [claim],
        "why_it_matters": [claim],
        "cross_source_comparison": {
            "agreements": [],
            "differences": [],
            "missing_information": []
        },
        "ai_generated_analysis": "This analysis is interpretation, not a reported fact.",
        "confidence": 0.81
    }
    script = {
        "suggested_title": "What this AI launch means",
        "thumbnail_text": "NEW AI MODEL",
        "target_duration_seconds": 75,
        "opening_hook": "A new AI announcement raises an important question.",
        "introduction": "Here is what the source reports.",
        "main_explanation": "The announcement describes the model and its details.",
        "important_developments": [
            sourced_claim(f"Development {index} is reported.", urls)
            for index in range(1, 4)
        ],
        "practical_examples": [claim],
        "why_viewers_should_care": "The source describes a possible practical use.",
        "closing": "That is the update, with the source linked below.",
        "source_urls": list(urls),
        "scenes": [
            {
                "scene_number": index,
                "duration_seconds": 15,
                "narration": f"Scene {index} explains a sourced detail.",
                "visual_suggestion": f"Show a simple graphic for scene {index}.",
                "on_screen_text": f"Detail {index}",
                "source_urls": list(urls)
            }
            for index in range(1, 6)
        ]
    }
    return {"research_brief": brief, "video_script": script}


def research_group(url=SOURCE_URL, content_source="full_article"):
    return {
        "topic": "AI model launch",
        "research_scope": "single_source",
        "sources": [
            {
                "publisher": "Example Publisher",
                "title": "AI model launch",
                "url": url,
                "canonical_url": url,
                "published": "2026-11-02",
                "author": "Example Author",
                "content_source": content_source,
                "content": (
                    "The source describes an AI model launch and its "
                    "reported details, applications, and limitations."
                )
            }
        ],
        "cross_check": {"claim_agreement": "Not assessed"}
    }


class FakeResearchClient:
    model = "gemini-test"
    max_output_tokens = 1200

    def __init__(self, response):
        self.response = response
        self.calls = []

    def generate_research_output(self, prompt):
        self.calls.append(prompt)
        return self.response


class FakeMessageResponse:
    def __init__(self, text):
        self.text = text


class FakeModels:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


class FakeSDKClient:
    def __init__(self, response):
        self.models = FakeModels(response)


def test_research_brief_and_video_output_validate_sources_and_timing():
    validated = validate_research_output(
        generated_output(),
        {SOURCE_URL}
    )

    assert len(validated["research_brief"]["important_developments"]) == 3
    assert validated["research_brief"]["confidence"] == 0.81
    assert validated["video_script"]["target_duration_seconds"] == 75
    assert sum(
        scene["duration_seconds"]
        for scene in validated["video_script"]["scenes"]
    ) == 75
    assert all(
        scene["source_urls"] == [SOURCE_URL]
        for scene in validated["video_script"]["scenes"]
    )


def test_research_output_rejects_unverified_urls_and_incomplete_developments():
    output = generated_output()
    output["research_brief"]["what_happened"][0]["source_urls"] = [
        "https://unknown.example/fact"
    ]

    with pytest.raises(ValueError, match="unrecognized source URL"):
        validate_research_output(output, {SOURCE_URL})

    output = generated_output()
    output["research_brief"]["important_developments"] = [
        output["research_brief"]["important_developments"][0]
    ]
    with pytest.raises(ValueError, match="3 to 5"):
        validate_research_output(output, {SOURCE_URL})


def test_research_output_rejects_scene_timing_outside_target():
    output = generated_output()
    output["video_script"]["scenes"][0]["duration_seconds"] = 5

    with pytest.raises(ValueError, match="durations must total"):
        validate_research_output(output, {SOURCE_URL})


def test_cross_source_comparison_requires_two_sources_for_claims():
    output = generated_output()
    output["research_brief"]["cross_source_comparison"]["agreements"] = [
        sourced_claim("Both reports describe the same announcement.")
    ]
    with pytest.raises(ValueError, match="single-source"):
        validate_research_output(output, {SOURCE_URL}, source_count=1)

    output = generated_output((SOURCE_URL, SECOND_SOURCE_URL))
    output["research_brief"]["cross_source_comparison"]["agreements"] = [
        sourced_claim(
            "Both reports describe the same announcement.",
            (SOURCE_URL, SECOND_SOURCE_URL)
        )
    ]
    validated = validate_research_output(
        output,
        {SOURCE_URL, SECOND_SOURCE_URL},
        source_count=2
    )
    assert len(
        validated["research_brief"]["cross_source_comparison"]["agreements"]
    ) == 1


def test_cross_source_comparison_cannot_count_canonical_aliases_as_publishers():
    output = generated_output(("https://example.com/original", "https://example.com/canonical"))
    output["research_brief"]["cross_source_comparison"]["agreements"] = [
        sourced_claim(
            "The reports agree.",
            ("https://example.com/original", "https://example.com/canonical")
        )
    ]

    with pytest.raises(ValueError, match="two sources"):
        validate_research_output(
            output,
            {
                "https://example.com/original",
                "https://example.com/canonical"
            },
            source_count=2,
            source_publishers={
                "https://example.com/original": "example",
                "https://example.com/canonical": "example"
            }
        )


def test_research_generation_marks_single_source_and_content_quality(tmp_path):
    client = FakeResearchClient(generated_output())
    cache = SummaryCache(tmp_path / "cache.json")

    results = generate_research_briefs(
        [research_group(content_source="rss_fallback")],
        client=client,
        cache=cache
    )

    assert results[0]["research_status"] == "SUCCESS"
    assert results[0]["research_scope"] == "single_source"
    assert results[0]["source_count"] == 1
    assert results[0]["source_urls"] == [SOURCE_URL]
    assert results[0]["content_quality"] == {
        "label": "source_summary_fallback",
        "sources_with_full_articles": 0,
        "sources_using_fallback": 1,
        "sources_without_content": 0
    }
    assert "ai_generated_analysis" in results[0]["research_brief"]
    assert "scenes" in results[0]["video_script"]
    assert len(client.calls) == 1


def test_research_generation_labels_multiple_publishers_and_keeps_citations(
    tmp_path
):
    group = research_group()
    group["research_scope"] = "multi_source"
    group["sources"].append({
        "publisher": "AI News",
        "title": "Another report on the AI model launch",
        "url": SECOND_SOURCE_URL,
        "canonical_url": SECOND_SOURCE_URL,
        "published": "2026-11-03",
        "author": "",
        "content_source": "rss_fallback",
        "content": "The outlet also describes the reported launch."
    })
    client = FakeResearchClient(
        generated_output((SOURCE_URL, SECOND_SOURCE_URL))
    )

    results = generate_research_briefs(
        [group],
        client=client,
        cache=SummaryCache(tmp_path / "cache.json")
    )

    assert results[0]["research_scope"] == "multi_source"
    assert results[0]["source_count"] == 2
    assert results[0]["source_urls"] == [SOURCE_URL, SECOND_SOURCE_URL]
    assert results[0]["content_quality"]["label"] == (
        "mixed_full_article_and_fallback"
    )
    assert results[0]["research_brief"]["what_happened"][0]["source_urls"] == [
        SOURCE_URL,
        SECOND_SOURCE_URL
    ]


def test_research_generation_uses_cache_for_unchanged_sources(tmp_path):
    client = FakeResearchClient(generated_output())
    cache = SummaryCache(tmp_path / "cache.json")
    group = research_group()

    first = generate_research_briefs([group], client=client, cache=cache)
    second = generate_research_briefs(
        [group],
        client=client,
        cache=SummaryCache(tmp_path / "cache.json")
    )

    assert first == second
    assert len(client.calls) == 1


def test_research_generation_rejects_unverifiable_model_output():
    output = generated_output()
    output["video_script"]["scenes"][0]["source_urls"] = [
        "https://fake.example/"
    ]
    client = FakeResearchClient(output)

    result = generate_research_briefs(
        [research_group()],
        client=client,
        cache=SummaryCache("/invalid/test/path")
    )

    assert result[0]["research_status"] == "FAILED"
    assert "known source URLs" in result[0]["error"]


def test_gemini_research_method_uses_sdk_structured_json_schema():
    response_text = json.dumps(generated_output())
    sdk = FakeSDKClient(FakeMessageResponse(response_text))
    client = GeminiClient("gemini-test-key", sdk_client=sdk)

    result = client.generate_research_output("Use these source facts.")

    request = sdk.models.calls[0]
    assert result == generated_output()
    assert request["config"]["response_mime_type"] == "application/json"
    schema = request["config"]["response_json_schema"]
    assert "research_brief" in schema["properties"]
    assert "video_script" in schema["properties"]
    assert "response_schema" not in request["config"]
    assert "additional_properties" not in json.dumps(schema)
    assert "additionalProperties" not in json.dumps(schema)
    assert request["config"]["max_output_tokens"] == 8192
    assert request["config"]["thinking_config"] == {
        "thinking_level": "LOW"
    }


def test_research_json_schema_serializes_with_sdk_response_json_schema():
    from google.genai import types

    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_json_schema=RESEARCH_OUTPUT_JSON_SCHEMA
    )
    serialized = config.model_dump(exclude_none=True, by_alias=True)
    encoded_schema = json.dumps(serialized["responseJsonSchema"])

    assert "responseJsonSchema" in serialized
    assert "additional_properties" not in encoded_schema
    assert "additionalProperties" not in encoded_schema
