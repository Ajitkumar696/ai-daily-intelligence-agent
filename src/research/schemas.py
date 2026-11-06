RESEARCH_CLAIM_SCHEMA = {
    "type": "object",
    "properties": {
        "statement": {"type": "string"},
        "source_urls": {
            "type": "array",
            "items": {"type": "string"}
        }
    },
    "required": ["statement", "source_urls"]
}

_CLAIM_ARRAY = {
    "type": "array",
    "items": RESEARCH_CLAIM_SCHEMA
}

_STRING_ARRAY = {
    "type": "array",
    "items": {"type": "string"}
}

_MISSING_INFORMATION_SCHEMA = {
    "type": "object",
    "properties": {
        "detail": {"type": "string"},
        "source_urls": _STRING_ARRAY
    },
    "required": ["detail", "source_urls"]
}

_SCENE_SCHEMA = {
    "type": "object",
    "properties": {
        "scene_number": {"type": "integer"},
        "duration_seconds": {"type": "integer"},
        "narration": {"type": "string"},
        "visual_suggestion": {"type": "string"},
        "on_screen_text": {"type": "string"},
        "source_urls": _STRING_ARRAY
    },
    "required": [
        "scene_number",
        "duration_seconds",
        "narration",
        "visual_suggestion",
        "on_screen_text",
        "source_urls"
    ]
}

RESEARCH_OUTPUT_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "research_brief": {
            "type": "object",
            "properties": {
                "background": _CLAIM_ARRAY,
                "what_happened": _CLAIM_ARRAY,
                "technology_explanation": _CLAIM_ARRAY,
                "important_developments": _CLAIM_ARRAY,
                "technical_details": _CLAIM_ARRAY,
                "companies_models_technologies": _CLAIM_ARRAY,
                "real_world_applications": _CLAIM_ARRAY,
                "benefits": _CLAIM_ARRAY,
                "limitations": _CLAIM_ARRAY,
                "potential_impact": _CLAIM_ARRAY,
                "why_it_matters": _CLAIM_ARRAY,
                "cross_source_comparison": {
                    "type": "object",
                    "properties": {
                        "agreements": _CLAIM_ARRAY,
                        "differences": _CLAIM_ARRAY,
                        "missing_information": {
                            "type": "array",
                            "items": _MISSING_INFORMATION_SCHEMA
                        }
                    },
                    "required": [
                        "agreements",
                        "differences",
                        "missing_information"
                    ]
                },
                "ai_generated_analysis": {"type": "string"},
                "confidence": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1
                }
            },
            "required": [
                "background",
                "what_happened",
                "technology_explanation",
                "important_developments",
                "technical_details",
                "companies_models_technologies",
                "real_world_applications",
                "benefits",
                "limitations",
                "potential_impact",
                "why_it_matters",
                "cross_source_comparison",
                "ai_generated_analysis",
                "confidence"
            ]
        },
        "video_script": {
            "type": "object",
            "properties": {
                "suggested_title": {"type": "string"},
                "thumbnail_text": {"type": "string"},
                "target_duration_seconds": {"type": "integer"},
                "opening_hook": {"type": "string"},
                "introduction": {"type": "string"},
                "main_explanation": {"type": "string"},
                "important_developments": _CLAIM_ARRAY,
                "practical_examples": _CLAIM_ARRAY,
                "why_viewers_should_care": {"type": "string"},
                "closing": {"type": "string"},
                "source_urls": _STRING_ARRAY,
                "scenes": {
                    "type": "array",
                    "items": _SCENE_SCHEMA
                }
            },
            "required": [
                "suggested_title",
                "thumbnail_text",
                "target_duration_seconds",
                "opening_hook",
                "introduction",
                "main_explanation",
                "important_developments",
                "practical_examples",
                "why_viewers_should_care",
                "closing",
                "source_urls",
                "scenes"
            ]
        }
    },
    "required": ["research_brief", "video_script"]
}

BRIEF_CLAIM_FIELDS = (
    "background",
    "what_happened",
    "technology_explanation",
    "important_developments",
    "technical_details",
    "companies_models_technologies",
    "real_world_applications",
    "benefits",
    "limitations",
    "potential_impact",
    "why_it_matters"
)


def _validate_claims(claims, field, known_urls, required=False):
    if not isinstance(claims, list):
        raise ValueError(f"{field} must be a list of sourced claims.")
    if required and not claims:
        raise ValueError(f"{field} must contain at least one sourced claim.")
    validated = []
    for claim in claims:
        if not isinstance(claim, dict):
            raise ValueError(f"{field} contains an invalid claim.")
        statement = claim.get("statement")
        urls = claim.get("source_urls")
        if not isinstance(statement, str) or not statement.strip():
            raise ValueError(f"{field} contains an empty claim.")
        if not isinstance(urls, list) or not urls:
            raise ValueError(f"{field} claims must cite at least one source URL.")
        if any(
            not isinstance(url, str) or url not in known_urls
            for url in urls
        ):
            raise ValueError(f"{field} contains an unrecognized source URL.")
        validated.append({
            "statement": statement.strip(),
            "source_urls": list(dict.fromkeys(urls))
        })
    return validated


def _validate_string(value, field):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string.")
    return value.strip()


def validate_research_output(
    data,
    known_urls,
    source_count=None,
    source_publishers=None,
    trusted_topic=None,
    trusted_headline=None
):
    """Validate generated brief claims, citations, video scenes, and timing."""

    if not isinstance(data, dict):
        raise ValueError("Research output must be a JSON object.")
    brief = data.get("research_brief")
    script = data.get("video_script")
    if not isinstance(brief, dict) or not isinstance(script, dict):
        raise ValueError("Research output must include a brief and video script.")
    if not known_urls:
        raise ValueError("Research output cannot be validated without sources.")

    result_brief = {
        "topic": _validate_string(
            trusted_topic if trusted_topic is not None else brief.get("topic"),
            "topic"
        ),
        "headline": _validate_string(
            trusted_headline if trusted_headline is not None else brief.get("headline"),
            "headline"
        )
    }
    for field in BRIEF_CLAIM_FIELDS:
        result_brief[field] = _validate_claims(
            brief.get(field),
            field,
            known_urls,
            required=field in ("what_happened", "important_developments")
        )
    if not 3 <= len(result_brief["important_developments"]) <= 5:
        raise ValueError(
            "important_developments must contain 3 to 5 sourced claims."
        )

    comparison = brief.get("cross_source_comparison")
    if not isinstance(comparison, dict):
        raise ValueError("cross_source_comparison must be an object.")
    source_count = source_count if source_count is not None else len(known_urls)
    validated_comparison = {}
    for field in ("agreements", "differences"):
        claims = _validate_claims(
            comparison.get(field),
            f"cross_source_comparison.{field}",
            known_urls
        )
        if source_count < 2 and claims:
            raise ValueError(
                f"{field} cannot claim cross-source comparison for a "
                "single-source topic."
            )
        if any(
            len({
                source_publishers.get(url, url)
                if source_publishers is not None else url
                for url in claim["source_urls"]
            }) < 2
            for claim in claims
        ):
            raise ValueError(
                f"cross_source_comparison.{field} claims must cite at least "
                "two sources."
            )
        validated_comparison[field] = claims

    missing_information = comparison.get("missing_information")
    if not isinstance(missing_information, list):
        raise ValueError("missing_information must be a list.")
    validated_missing = []
    for item in missing_information:
        if not isinstance(item, dict):
            raise ValueError("missing_information contains an invalid entry.")
        detail = item.get("detail")
        urls = item.get("source_urls")
        if not isinstance(detail, str) or not detail.strip():
            raise ValueError("missing_information details must be non-empty.")
        if (
            not isinstance(urls, list)
            or not urls
            or any(
                not isinstance(url, str) or url not in known_urls
                for url in urls
            )
        ):
            raise ValueError(
                "missing_information entries must cite known source URLs."
            )
        validated_missing.append({
            "detail": detail.strip(),
            "source_urls": list(dict.fromkeys(urls))
        })
    validated_comparison["missing_information"] = validated_missing
    result_brief["cross_source_comparison"] = validated_comparison

    analysis = brief.get("ai_generated_analysis")
    if not isinstance(analysis, str):
        raise ValueError("ai_generated_analysis must be a string.")
    confidence = brief.get("confidence")
    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not 0 <= confidence <= 1
    ):
        raise ValueError("confidence must be a number from 0 to 1.")
    result_brief["ai_generated_analysis"] = analysis.strip()
    result_brief["confidence"] = float(confidence)

    result_script = {
        field: _validate_string(script.get(field), field)
        for field in (
            "suggested_title",
            "thumbnail_text",
            "opening_hook",
            "introduction",
            "main_explanation",
            "why_viewers_should_care",
            "closing"
        )
    }
    duration = script.get("target_duration_seconds")
    if isinstance(duration, bool) or not isinstance(duration, int):
        raise ValueError("target_duration_seconds must be an integer.")
    if not 60 <= duration <= 90:
        raise ValueError("Video target duration must be from 60 to 90 seconds.")
    result_script["target_duration_seconds"] = duration
    result_script["important_developments"] = _validate_claims(
        script.get("important_developments"),
        "video important_developments",
        known_urls,
        required=True
    )
    if not 3 <= len(result_script["important_developments"]) <= 5:
        raise ValueError(
            "Video important_developments must contain 3 to 5 claims."
        )
    result_script["practical_examples"] = _validate_claims(
        script.get("practical_examples"),
        "video practical_examples",
        known_urls
    )

    script_urls = script.get("source_urls")
    if (
        not isinstance(script_urls, list)
        or not script_urls
        or any(
            not isinstance(url, str) or url not in known_urls
            for url in script_urls
        )
    ):
        raise ValueError("Video script must cite known source URLs.")
    result_script["source_urls"] = list(dict.fromkeys(script_urls))

    scenes = script.get("scenes")
    if not isinstance(scenes, list) or not 4 <= len(scenes) <= 8:
        raise ValueError("Video script must contain 4 to 8 timed scenes.")
    result_scenes = []
    scene_durations = 0
    for expected_number, scene in enumerate(scenes, start=1):
        if not isinstance(scene, dict):
            raise ValueError("Video script contains an invalid scene.")
        number = scene.get("scene_number")
        scene_duration = scene.get("duration_seconds")
        if (
            isinstance(number, bool)
            or not isinstance(number, int)
            or number != expected_number
        ):
            raise ValueError("Video scene numbers must be sequential.")
        if (
            isinstance(scene_duration, bool)
            or not isinstance(scene_duration, int)
            or scene_duration <= 0
        ):
            raise ValueError("Every video scene needs a positive duration.")
        urls = scene.get("source_urls")
        if (
            not isinstance(urls, list)
            or not urls
            or any(
                not isinstance(url, str) or url not in known_urls
                for url in urls
            )
        ):
            raise ValueError("Every video scene must cite known source URLs.")
        result_scenes.append({
            "scene_number": number,
            "duration_seconds": scene_duration,
            "narration": _validate_string(
                scene.get("narration"), "scene narration"
            ),
            "visual_suggestion": _validate_string(
                scene.get("visual_suggestion"), "scene visual_suggestion"
            ),
            "on_screen_text": _validate_string(
                scene.get("on_screen_text"), "scene on_screen_text"
            ),
            "source_urls": list(dict.fromkeys(urls))
        })
        scene_durations += scene_duration

    if abs(scene_durations - duration) > 5:
        raise ValueError(
            "Timed scene durations must total within five seconds of the "
            "video target duration."
        )
    result_script["scenes"] = result_scenes

    return {
        "research_brief": result_brief,
        "video_script": result_script
    }
