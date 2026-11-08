import pytest

import backend.rehab_audio.text_analysis as text_analysis
from backend.rehab_audio.text_analysis import AzureLanguageClient


def test_analyze_parses_sentiment_phrases_and_entities(monkeypatch, fake_response):
    def fake_post(url, params=None, headers=None, json=None, timeout=None):
        kind = json["kind"]
        if kind == "SentimentAnalysis":
            return fake_response(
                {
                    "results": {
                        "documents": [
                            {
                                "sentiment": "negative",
                                "confidenceScores": {
                                    "positive": 0.1,
                                    "neutral": 0.2,
                                    "negative": 0.7,
                                },
                            }
                        ]
                    }
                }
            )
        if kind == "KeyPhraseExtraction":
            return fake_response(
                {
                    "results": {
                        "documents": [
                            {"keyPhrases": ["dor no joelho", "cansaco"]}
                        ]
                    }
                }
            )
        return fake_response(
            {
                "results": {
                    "documents": [
                        {
                            "entities": [
                                {"text": "joelho", "category": "Anatomy"}
                            ]
                        }
                    ]
                }
            }
        )

    monkeypatch.setattr(text_analysis.requests, "post", fake_post)

    result = AzureLanguageClient("key", "https://example.cognitiveservices.azure.com/").analyze(
        "sinto dor no joelho e cansaco"
    )

    assert result["sentiment"] == {
        "label": "negative",
        "positive": 0.1,
        "neutral": 0.2,
        "negative": 0.7,
    }
    assert result["key_phrases"] == ["dor no joelho", "cansaco"]
    assert result["entities"] == [{"text": "joelho", "category": "Anatomy"}]


def test_analyze_empty_text_uses_neutral_defaults_without_network(monkeypatch):
    def fail_post(*args, **kwargs):
        raise AssertionError("network should not be called for empty text")

    monkeypatch.setattr(text_analysis.requests, "post", fail_post)

    result = AzureLanguageClient("key", "https://example.cognitiveservices.azure.com").analyze(
        "   "
    )

    assert result["sentiment"] == {
        "label": "neutral",
        "positive": 0.0,
        "neutral": 1.0,
        "negative": 0.0,
    }
    assert result["key_phrases"] == []
    assert result["entities"] == []


def test_analyze_raises_runtime_error_on_non_200(monkeypatch, fake_response):
    monkeypatch.setattr(
        text_analysis.requests,
        "post",
        lambda *args, **kwargs: fake_response({}, status_code=403, text="forbidden"),
    )

    with pytest.raises(RuntimeError, match="403"):
        AzureLanguageClient("key", "https://example.cognitiveservices.azure.com").analyze(
            "texto"
        )
