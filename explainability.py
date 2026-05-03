"""Gemini-backed explanations for model predictions.

This module intentionally reads the API key from the environment at call time so
the `/configure` endpoint can enable explanations without restarting the app.
"""

import logging
import os
from typing import Optional

from google import genai
from PIL import Image


logger = logging.getLogger("Explainability")

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")


def is_gemini_configured() -> bool:
    """Return whether a Gemini API key is available for this process."""
    return bool(os.getenv("GEMINI_API_KEY", "").strip())


def generate_explanation(prediction: str, context: Optional[Image.Image] = None) -> str:
    """
    Generate a short natural-language explanation for a model prediction.

    The optional image context parameter is reserved for future multimodal
    explanations; the current local app uses text-only explanations for speed
    and predictable behavior.
    """
    del context

    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        return "Explanation unavailable: Gemini API key is not configured."

    safe_prediction = (prediction or "").strip()
    if not safe_prediction:
        return "Explanation unavailable: no prediction was provided."

    if safe_prediction.lower() == "uncertain":
        return (
            "The model confidence was below the configured threshold, so the "
            "system did not report a specific action."
        )

    prompt = (
        f"An AI video model predicted the action '{safe_prediction}' in a short clip. "
        "Explain in 1-2 concise sentences what this action usually looks like "
        "and what motion cues a video model may have used. Avoid claiming that "
        "the prediction is certainly correct."
    )

    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
        )
        text = (response.text or "").strip()
        return text or "Explanation unavailable: Gemini returned an empty response."
    except Exception as exc:
        logger.warning("Gemini explanation failed: %s", exc)
        return "Explanation unavailable: Gemini could not generate a response right now."


if __name__ == "__main__":
    print(generate_explanation("punch"))
