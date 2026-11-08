"""Opt-in live tests against real Azure Speech and Azure AI Language.

These tests are skipped unless the matching credentials are present in the
environment, so the default suite stays fully offline. To run them once real
credentials are configured:

    $env:AZURE_SPEECH_KEY="..."; $env:AZURE_SPEECH_REGION="brazilsouth"
    $env:AZURE_LANGUAGE_KEY="..."; $env:AZURE_LANGUAGE_ENDPOINT="https://<resource>.cognitiveservices.azure.com"
    $env:AZURE_AUDIO_SAMPLE_FILE="entregas/fase04/data/raw/primock57/day1_consultation01_patient_60s.wav"
    $env:REHAB_AUDIO_LANGUAGE="en-GB"  # PriMock57 e em ingles; padrao do projeto e pt-BR
    python -m pytest tests/fase04/test_azure_audio_live.py -q

No audio or transcript is written to disk or logged (privacy constraint).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from backend.rehab_audio.text_analysis import AzureLanguageClient
from backend.rehab_audio.transcription import AzureSpeechClient


def _live_language() -> str:
    # Padrao pt-BR (alvo do projeto); use REHAB_AUDIO_LANGUAGE=en-GB para
    # validar a transcricao com o PriMock57, que e em ingles.
    return os.getenv("REHAB_AUDIO_LANGUAGE", "pt-BR")


def _has_speech_credentials() -> bool:
    return bool(os.getenv("AZURE_SPEECH_KEY") and os.getenv("AZURE_SPEECH_REGION"))


def _has_language_credentials() -> bool:
    return bool(
        os.getenv("AZURE_LANGUAGE_KEY") and os.getenv("AZURE_LANGUAGE_ENDPOINT")
    )


requires_speech = pytest.mark.skipif(
    not _has_speech_credentials(),
    reason="AZURE_SPEECH_KEY/AZURE_SPEECH_REGION not set",
)
requires_language = pytest.mark.skipif(
    not _has_language_credentials(),
    reason="AZURE_LANGUAGE_KEY/AZURE_LANGUAGE_ENDPOINT not set",
)


@requires_language
def test_live_language_analysis_returns_sentiment():
    client = AzureLanguageClient(
        os.environ["AZURE_LANGUAGE_KEY"],
        os.environ["AZURE_LANGUAGE_ENDPOINT"],
        language="pt-BR",
    )

    result = client.analyze("Estou com muita dor no joelho e cansaco.")

    assert result["sentiment"]["label"] in {"positive", "neutral", "negative"}
    assert isinstance(result["key_phrases"], list)
    assert isinstance(result["entities"], list)


@requires_speech
def test_live_transcription_returns_transcript():
    sample = os.getenv("AZURE_AUDIO_SAMPLE_FILE")
    if not sample or not Path(sample).is_file():
        pytest.skip(
            "Set AZURE_AUDIO_SAMPLE_FILE to a .wav file to run live transcription."
        )

    client = AzureSpeechClient(
        os.environ["AZURE_SPEECH_KEY"],
        os.environ["AZURE_SPEECH_REGION"],
        language=_live_language(),
    )

    result = client.transcribe(Path(sample))

    assert isinstance(result["transcript"], str)
    assert result["language"] == _live_language()
