import pytest
import requests

from src.collectors import article_extractor
from src.collectors.article_extractor import _LAST_REQUEST_AT, _ROBOTS_CACHE


@pytest.fixture(autouse=True)
def prevent_network_requests(monkeypatch):
    def deny_request(*args, **kwargs):
        pytest.fail("Automated tests must not make network requests.")

    monkeypatch.setattr(
        requests.sessions.Session,
        "request",
        deny_request
    )
    _ROBOTS_CACHE.clear()
    _LAST_REQUEST_AT.clear()
    monkeypatch.setattr(article_extractor, "MIN_REQUEST_INTERVAL_SECONDS", 0)
    yield
    _ROBOTS_CACHE.clear()
    _LAST_REQUEST_AT.clear()
