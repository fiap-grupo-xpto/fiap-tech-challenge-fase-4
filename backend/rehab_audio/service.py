from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import soundfile as sf

from backend.rehab_audio.acoustics import (
    TARGET_FEATURE_SAMPLE_RATE,
    acoustic_findings,
    decimate,
    extract_acoustic_features,
    to_mono,
)
from backend.rehab_audio.report import build_support_report
from backend.rehab_audio.terms import extract_critical_terms
from backend.rehab_audio.text_analysis import AzureLanguageClient
from backend.rehab_audio.transcription import AzureSpeechClient
from backend.rehab_audio.voice_model import predict_voice_alteration


DEFAULT_LANGUAGE = os.getenv("REHAB_AUDIO_LANGUAGE", "pt-BR")
MAX_AUDIO_SECONDS = int(os.getenv("MAX_REHAB_AUDIO_SECONDS", "60"))
# The Azure short-audio REST endpoint only accepts WAV (PCM, 16 kHz, mono).
AZURE_STT_SAMPLE_RATE = 16000
REHAB_PHRASE_LIST = [
    "inspire",
    "expire",
    "repita o movimento",
    "amplitude",
    "descanse",
]

_MISSING_CREDENTIALS = (
    "Credenciais do Azure nao configuradas para analise de audio."
)


@lru_cache(maxsize=1)
def _get_stt_client() -> AzureSpeechClient:
    key = os.getenv("AZURE_SPEECH_KEY")
    region = os.getenv("AZURE_SPEECH_REGION")
    if not key or not region:
        raise RuntimeError(_MISSING_CREDENTIALS)
    return AzureSpeechClient(
        key,
        region,
        language=DEFAULT_LANGUAGE,
        phrase_list=REHAB_PHRASE_LIST,
    )


@lru_cache(maxsize=1)
def _get_language_client() -> AzureLanguageClient:
    key = os.getenv("AZURE_LANGUAGE_KEY")
    endpoint = os.getenv("AZURE_LANGUAGE_ENDPOINT")
    if not key or not endpoint:
        raise RuntimeError(_MISSING_CREDENTIALS)
    return AzureLanguageClient(key, endpoint, language=DEFAULT_LANGUAGE)


def analyze_session_audio(
    audio_path: Path | str,
    language: str | None = None,
) -> dict:
    data, sample_rate = sf.read(str(audio_path), dtype="float32", always_2d=False)
    duration = len(data) / sample_rate if sample_rate else 0.0
    if duration > MAX_AUDIO_SECONDS:
        raise ValueError(
            f"Audio exceeds the {MAX_AUDIO_SECONDS} second limit for this analysis."
        )

    channels = data.shape[1] if data.ndim > 1 else 1
    if sample_rate != AZURE_STT_SAMPLE_RATE or channels > 1:
        raise ValueError(
            "O Azure Speech (audio curto) exige WAV PCM 16 kHz mono; "
            f"recebido {sample_rate} Hz com {channels} canal(is)."
        )

    speech_client = _get_stt_client()
    if language and language != speech_client.language:
        speech_client = AzureSpeechClient(
            speech_client.key,
            speech_client.region,
            language=language,
            phrase_list=REHAB_PHRASE_LIST,
        )
    transcription = speech_client.transcribe(Path(audio_path))

    transcript = transcription.get("transcript", "")
    text_analysis = _get_language_client().analyze(transcript)

    signal, feature_rate = decimate(
        to_mono(data), int(sample_rate), TARGET_FEATURE_SAMPLE_RATE
    )
    metrics = extract_acoustic_features(signal, feature_rate)
    findings = acoustic_findings(metrics)
    terms = extract_critical_terms(transcript)
    voice_alteration = predict_voice_alteration(metrics)

    return build_support_report(
        transcription,
        text_analysis,
        metrics,
        findings,
        terms,
        voice_alteration,
    )
