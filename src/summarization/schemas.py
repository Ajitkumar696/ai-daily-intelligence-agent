import math


CATEGORIES = (
    "AI Models",
    "Research",
    "Generative AI",
    "AI Agents",
    "Developer Tools",
    "Open Source",
    "Business",
    "Industry News",
    "Robotics",
    "Other"
)

IMPORTANCE_LEVELS = ("High", "Medium", "Low")

SUMMARY_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "category": {"type": "string", "enum": list(CATEGORIES)},
        "short_summary": {"type": "string"},
        "key_points": {
            "type": "array",
            "items": {"type": "string"}
        },
        "why_it_matters": {"type": "string"},
        "importance": {
            "type": "string",
            "enum": list(IMPORTANCE_LEVELS)
        },
        "confidence": {
            "type": "number",
            "minimum": 0,
            "maximum": 1
        }
    },
    "required": [
        "category",
        "short_summary",
        "key_points",
        "why_it_matters",
        "importance",
        "confidence"
    ]
}

GENERATED_FIELDS = tuple(SUMMARY_JSON_SCHEMA["required"])


def validate_summary_data(data, trusted_title=None):
    """Validate Claude's structured response before it enters the output."""

    if not isinstance(data, dict):
        raise ValueError("Claude response must be a JSON object.")

    missing = set(GENERATED_FIELDS) - data.keys()
    if missing:
        raise ValueError(
            "Claude response is missing required fields: "
            + ", ".join(sorted(missing))
        )

    extra = data.keys() - set(GENERATED_FIELDS) - {"title"}
    if extra:
        raise ValueError(
            "Claude response contains unsupported fields: "
            + ", ".join(sorted(extra))
        )

    title = trusted_title if trusted_title is not None else data.get("title")
    if not isinstance(title, str) or not title.strip():
        raise ValueError("Trusted article title must be non-empty.")
    for field in ("short_summary", "why_it_matters"):
        if not isinstance(data[field], str) or not data[field].strip():
            raise ValueError(f"Claude response field {field!r} must be non-empty.")

    if len(data["short_summary"]) > 700:
        raise ValueError("Claude short_summary exceeds 700 characters.")

    if data["category"] not in CATEGORIES:
        raise ValueError("Claude response contains an unsupported category.")

    if data["importance"] not in IMPORTANCE_LEVELS:
        raise ValueError("Claude response contains an unsupported importance.")

    points = data["key_points"]
    if (
        not isinstance(points, list)
        or not 1 <= len(points) <= 6
        or any(not isinstance(point, str) or not point.strip() for point in points)
    ):
        raise ValueError("Claude key_points must contain 1 to 6 non-empty strings.")

    confidence = data["confidence"]
    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not math.isfinite(confidence)
        or not 0 <= confidence <= 1
    ):
        raise ValueError("Claude confidence must be a number from 0 to 1.")

    return {
        "title": title.strip(),
        "category": data["category"],
        "short_summary": data["short_summary"].strip(),
        "key_points": [point.strip() for point in points],
        "why_it_matters": data["why_it_matters"].strip(),
        "importance": data["importance"],
        "confidence": float(confidence)
    }
