"""Image generation tool for creating improved landing page designs.

Supports Google Gemini (native image generation) and LiteLLM-backed
providers (e.g. OpenAI DALL·E 3) as backends.
"""

import logging
import os
from datetime import datetime
from typing import Optional

import requests
from google import genai
from google.genai import types

from src.config.settings import settings

logger = logging.getLogger(__name__)


async def generate_improved_landing_page(
    prompt: str,
    reference_image_path: Optional[str] = None,
    output_dir: str = "artifacts",
) -> str:
    """Generate an improved landing page image and return its file path.

    Delegates to the Gemini native API or LiteLLM depending on the
    configured ``IMAGE_GENERATION_MODEL``.
    """
    os.makedirs(output_dir, exist_ok=True)
    filepath = _build_output_path(output_dir)

    logger.info("Generating image — prompt length: %d chars", len(prompt))

    if "gemini" in settings.IMAGE_GENERATION_MODEL.lower():
        return _generate_with_gemini(prompt, reference_image_path, filepath)

    return _generate_with_litellm(prompt, filepath)


# ── Private helpers ─────────────────────────────────────────────────


def _build_output_path(output_dir: str) -> str:
    """Return a timestamped PNG file path inside *output_dir*."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(output_dir, f"improved_design_{timestamp}.png")


def _generate_with_gemini(
    prompt: str,
    reference_image_path: Optional[str],
    filepath: str,
) -> str:
    """Use the Google GenAI SDK for image generation / editing."""
    client = genai.Client(api_key=settings.GEMINI_API_KEY)

    content_parts = [types.Part.from_text(text=prompt)]

    if reference_image_path and os.path.exists(reference_image_path):
        with open(reference_image_path, "rb") as fh:
            image_bytes = fh.read()
        content_parts.append(
            types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
        )

    contents = [
        types.Content(role="user", parts=content_parts),
    ]
    config = types.GenerateContentConfig(
        response_modalities=["IMAGE", "TEXT"],
    )

    response = client.models.generate_content(
        model=settings.IMAGE_GENERATION_MODEL,
        contents=contents,
        config=config,
    )

    return _extract_image_from_response(response, filepath)


def _extract_image_from_response(response, filepath: str) -> str:
    """Write the first inline image from *response* to *filepath*."""
    if response.candidates:
        candidate = response.candidates[0]
        if candidate.content and candidate.content.parts:
            for part in candidate.content.parts:
                if part.inline_data and part.inline_data.data:
                    with open(filepath, "wb") as fh:
                        fh.write(part.inline_data.data)
                    logger.info("Saved generated image to %s", filepath)
                    return filepath

    raise RuntimeError("Gemini model did not return an image.")


def _generate_with_litellm(prompt: str, filepath: str) -> str:
    """Fallback: use LiteLLM image_generation (e.g. OpenAI DALL·E 3)."""
    from litellm import image_generation  # noqa: E402 — conditional dep

    response = image_generation(
        prompt=prompt,
        model=settings.IMAGE_GENERATION_MODEL,
    )

    image_url = response.data[0].url
    img_data = requests.get(image_url, timeout=30).content
    with open(filepath, "wb") as fh:
        fh.write(img_data)

    logger.info("Saved generated image to %s", filepath)
    return filepath
