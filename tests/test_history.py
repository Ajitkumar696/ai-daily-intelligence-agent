import json

import pytest

import src.processing.history as history


def test_history_filters_and_marks_using_temporary_file(tmp_path):
    history_file = tmp_path / "processed_history.json"
    articles = [
        {"title": "New GPT Model", "url": "https://example.com/gpt"},
        {"title": "New Claude Model", "url": "https://example.com/claude"}
    ]

    assert history.load_history(history_file) == set()
    assert history.filter_previously_reported(
        articles,
        history_file=history_file
    ) == articles

    history.mark_as_reported([articles[0]], history_file=history_file)

    assert json.loads(history_file.read_text(encoding="utf-8")) == [
        "https://example.com/gpt"
    ]
    assert history.filter_previously_reported(
        articles,
        history_file=history_file
    ) == [articles[1]]


@pytest.mark.parametrize(
    "contents",
    ["{invalid", "{}", '["https://example.com", 12]', '[""]']
)
def test_history_rejects_malformed_or_invalid_json_structure(
    tmp_path,
    contents
):
    history_file = tmp_path / "processed_history.json"
    history_file.write_text(contents, encoding="utf-8")

    with pytest.raises(ValueError, match="history"):
        history.load_history(history_file)


def test_history_matches_urls_after_canonicalization(tmp_path):
    history_file = tmp_path / "processed_history.json"
    history_file.write_text(
        json.dumps(["https://EXAMPLE.com/article?utm_source=mail#top"]),
        encoding="utf-8"
    )
    article = {"url": "https://example.com/article"}

    assert history.filter_previously_reported(
        [article],
        history_file=history_file
    ) == []
