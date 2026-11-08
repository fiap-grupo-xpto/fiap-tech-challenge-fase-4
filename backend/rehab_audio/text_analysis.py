from __future__ import annotations

import requests


_NEUTRAL_SENTIMENT = {
    "label": "neutral",
    "positive": 0.0,
    "neutral": 1.0,
    "negative": 0.0,
}


def _first_document(payload: dict) -> dict:
    results = payload.get("results") or {}
    documents = results.get("documents") or []
    return documents[0] if documents else {}


def _parse_sentiment(payload: dict) -> dict:
    document = _first_document(payload)
    scores = document.get("confidenceScores") or {}
    return {
        "label": str(document.get("sentiment", "neutral")),
        "positive": float(scores.get("positive", 0.0) or 0.0),
        "neutral": float(scores.get("neutral", 0.0) or 0.0),
        "negative": float(scores.get("negative", 0.0) or 0.0),
    }


def _parse_key_phrases(payload: dict) -> list[str]:
    document = _first_document(payload)
    return [str(phrase) for phrase in (document.get("keyPhrases") or [])]


def _parse_entities(payload: dict) -> list[dict]:
    document = _first_document(payload)
    return [
        {
            "text": str(entity.get("text", "")),
            "category": str(entity.get("category", "")),
        }
        for entity in (document.get("entities") or [])
    ]


class AzureLanguageClient:
    def __init__(self, key: str, endpoint: str, language: str = "pt-BR") -> None:
        self.key = key
        self.endpoint = endpoint.rstrip("/")
        self.language = language

    def _post(self, kind: str, text: str) -> dict:
        response = requests.post(
            f"{self.endpoint}/language/:analyze-text",
            params={"api-version": "2023-04-01"},
            headers={
                "Ocp-Apim-Subscription-Key": self.key,
                "Content-Type": "application/json",
            },
            json={
                "kind": kind,
                "parameters": {"modelVersion": "latest"},
                "analysisInput": {
                    "documents": [
                        {"id": "1", "language": self.language, "text": text}
                    ]
                },
            },
            timeout=60,
        )
        if response.status_code != 200:
            raise RuntimeError(
                "Azure Language request failed with status "
                f"{response.status_code}: {response.text}"
            )
        return response.json()

    def analyze(self, text: str) -> dict:
        if not text.strip():
            return {
                "sentiment": dict(_NEUTRAL_SENTIMENT),
                "key_phrases": [],
                "entities": [],
            }

        sentiment_payload = self._post("SentimentAnalysis", text)
        phrases_payload = self._post("KeyPhraseExtraction", text)
        entities_payload = self._post("EntityRecognition", text)

        return {
            "sentiment": _parse_sentiment(sentiment_payload),
            "key_phrases": _parse_key_phrases(phrases_payload),
            "entities": _parse_entities(entities_payload),
        }
