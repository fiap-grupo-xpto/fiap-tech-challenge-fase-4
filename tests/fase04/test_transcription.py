import pytest

import backend.rehab_audio.transcription as transcription
from backend.rehab_audio.transcription import AzureSpeechClient


def test_transcribe_parses_word_segments(monkeypatch, tmp_path, fake_response):
    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"RIFFfakeaudio")
    payload = {
        "DisplayText": "repita o movimento",
        "Duration": 11_000_000,
        "NBest": [
            {
                "Confidence": 0.9,
                "Words": [
                    {
                        "Word": "repita",
                        "Offset": 0,
                        "Duration": 5_000_000,
                        "Confidence": 0.95,
                    },
                    {
                        "Word": "movimento",
                        "Offset": 5_000_000,
                        "Duration": 6_000_000,
                        "Confidence": 0.9,
                    },
                ],
            }
        ],
    }
    captured = {}

    def fake_post(url, params=None, headers=None, data=None, timeout=None):
        captured["url"] = url
        captured["headers"] = headers
        captured["data"] = data
        return fake_response(payload)

    monkeypatch.setattr(transcription.requests, "post", fake_post)

    result = AzureSpeechClient("secret-key", "brazilsouth").transcribe(audio_path)

    assert result["transcript"] == "repita o movimento"
    assert result["language"] == "pt-BR"
    assert len(result["segments"]) == 2
    assert result["segments"][0]["start"] == 0.0
    assert result["segments"][1]["start"] == 0.5
    assert result["segments"][1]["end"] == pytest.approx(1.1)
    assert captured["headers"]["Ocp-Apim-Subscription-Key"] == "secret-key"
    assert captured["url"].startswith("https://brazilsouth.")
    assert captured["data"] == b"RIFFfakeaudio"


def test_transcribe_falls_back_to_single_segment(monkeypatch, tmp_path, fake_response):
    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"data")
    payload = {"DisplayText": "texto completo", "Duration": 20_000_000}

    monkeypatch.setattr(
        transcription.requests,
        "post",
        lambda *args, **kwargs: fake_response(payload),
    )

    result = AzureSpeechClient("key", "region").transcribe(audio_path)

    assert len(result["segments"]) == 1
    assert result["segments"][0]["text"] == "texto completo"
    assert result["segments"][0]["end"] == 2.0


def test_transcribe_raises_runtime_error_on_non_200(monkeypatch, tmp_path, fake_response):
    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"data")

    monkeypatch.setattr(
        transcription.requests,
        "post",
        lambda *args, **kwargs: fake_response({}, status_code=401, text="unauthorized"),
    )

    with pytest.raises(RuntimeError, match="401"):
        AzureSpeechClient("key", "region").transcribe(audio_path)


def test_transcribe_sends_wav_content_type(monkeypatch, tmp_path, fake_response):
    payload = {"DisplayText": "texto", "Duration": 10_000_000}
    captured = {}

    def fake_post(url, params=None, headers=None, data=None, timeout=None):
        captured["headers"] = headers
        return fake_response(payload)

    monkeypatch.setattr(transcription.requests, "post", fake_post)

    wav_path = tmp_path / "sessao.wav"
    wav_path.write_bytes(b"data")
    AzureSpeechClient("key", "region").transcribe(wav_path)

    assert captured["headers"]["Content-Type"] == "audio/wav"
