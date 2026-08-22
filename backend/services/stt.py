"""
Speech-to-text service using Sarvam AI's Saarika model.
Docs: https://docs.sarvam.ai/api-reference-docs/speech-to-text/transcribe
"""
import os
import requests
from dotenv import load_dotenv

load_dotenv()

SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")
SARVAM_STT_URL = "https://api.sarvam.ai/speech-to-text"


class STTError(Exception):
    """Raised when transcription fails after retries."""
    pass


def transcribe_audio(file_bytes: bytes, filename: str = "audio.wav", language_code: str = "en-IN") -> dict:
    """
    Sends raw audio bytes to Sarvam's STT endpoint and returns the transcript.

    Returns:
        {"transcript": str, "language_code": str, "request_id": str | None}

    Raises:
        STTError if the request fails after retries or Sarvam returns an error.
    """
    if not SARVAM_API_KEY:
        raise STTError("SARVAM_API_KEY is not set in environment")

    headers = {"api-subscription-key": SARVAM_API_KEY}
    files = {"file": (filename, file_bytes, "audio/wav")}
    data = {
        "model": "saarika:v2.5",
        "language_code": language_code,  # "unknown" lets Sarvam auto-detect
    }

    last_exc = None
    for attempt in range(3):
        try:
            resp = requests.post(
                SARVAM_STT_URL, headers=headers, files=files, data=data, timeout=15
            )
            if resp.status_code == 429:
                # rate limited — brief backoff and retry
                import time
                time.sleep(1.5 * (attempt + 1))
                continue
            resp.raise_for_status()
            payload = resp.json()
            transcript = payload.get("transcript", "").strip()
            if not transcript:
                raise STTError("Sarvam returned an empty transcript")
            return {
                "transcript": transcript,
                "language_code": payload.get("language_code", language_code),
                "request_id": payload.get("request_id"),
            }
        except requests.RequestException as e:
            last_exc = e
            continue

    raise STTError(f"STT failed after retries: {last_exc}")
