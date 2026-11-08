from __future__ import annotations

from pathlib import Path

import requests


_API_URL_TEMPLATE = (
    "https://{region}.stt.speech.microsoft.com/"
    "speech/recognition/conversation/cognitiveservices/v1"
)
_TICKS_PER_SECOND = 10_000_000.0
# Only WAV (PCM) is accepted: the short-audio REST endpoint rejects MP3, which
# exists only for the fast/batch transcription API.
_CONTENT_TYPE = "audio/wav"


class AzureSpeechClient:
    def __init__(
        self,
        key: str,
        region: str,
        language: str = "pt-BR",
        phrase_list: list[str] | None = None,
    ) -> None:
        self.key = key
        self.region = region
        self.language = language
        self.phrase_list = list(phrase_list) if phrase_list else []

    def transcribe(self, audio_path: Path) -> dict:
        audio_path = Path(audio_path)
        response = requests.post(
            _API_URL_TEMPLATE.format(region=self.region),
            params={
                "language": self.language,
                "format": "detailed",
                "profanity": "raw",
            },
            headers={
                "Ocp-Apim-Subscription-Key": self.key,
                "Ocp-Apim-Subscription-Region": self.region,
                "Content-Type": _CONTENT_TYPE,
            },
            data=audio_path.read_bytes(),
            timeout=60,
        )
        if response.status_code != 200:
            raise RuntimeError(
                "Azure Speech request failed with status "
                f"{response.status_code}: {response.text}"
            )

        payload = response.json()
        transcript = payload.get("DisplayText") or ""
        segments = self._build_segments(payload, transcript)
        return {
            "transcript": transcript,
            "segments": segments,
            "language": self.language,
        }

    def _single_segment(
        self, text: str, end_seconds: float, confidence: float = 0.0
    ) -> dict:
        return {
            "start": 0.0,
            "end": round(end_seconds, 4),
            "text": text,
            "confidence": round(confidence, 4),
        }

    def _build_segments(self, payload: dict, transcript: str) -> list[dict]:
        duration_seconds = float(payload.get("Duration", 0) or 0) / _TICKS_PER_SECOND
        candidates = payload.get("NBest") or []
        if not candidates:
            return [self._single_segment(transcript, duration_seconds)]

        best = candidates[0]
        confidence = float(best.get("Confidence", 0.0) or 0.0)
        words = best.get("Words") or []
        if not words:
            return [self._single_segment(transcript, duration_seconds, confidence)]

        segments: list[dict] = []
        for word in words:
            start = float(word.get("Offset", 0) or 0) / _TICKS_PER_SECOND
            word_duration = float(word.get("Duration", 0) or 0) / _TICKS_PER_SECOND
            segments.append(
                {
                    "start": round(start, 4),
                    "end": round(start + word_duration, 4),
                    "text": str(word.get("Word", "")),
                    "confidence": round(
                        float(word.get("Confidence", confidence) or 0.0), 4
                    ),
                }
            )
        return segments
