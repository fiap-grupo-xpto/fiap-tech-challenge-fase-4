"""Shared fixtures for the fase04 test suite."""

import pytest


class FakeResponse:
    """Minimal stand-in for ``requests.Response`` used by the Azure client tests."""

    def __init__(self, payload, status_code=200, text=""):
        self._payload = payload
        self.status_code = status_code
        self.text = text

    def json(self):
        return self._payload


@pytest.fixture
def fake_response():
    return FakeResponse
