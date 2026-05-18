"""Unified LLM client for text, vision, and structured output calls.

Wraps LiteLLM's ``completion`` API with automatic model selection
(vision vs. text) based on whether an image is provided.
"""

import base64
import io
import logging
from typing import Any, Optional, Type, TypeVar

from PIL import Image
from pydantic import BaseModel

from litellm import completion
from src.config.settings import settings

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


def encode_image_to_base64(image_path: str) -> str:
    """Read an image file and return its Base64-encoded string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def _downscale_image(image_path: str, max_dim: int) -> str:
    """Downscale *image_path* so its longest side ≤ *max_dim*.

    Returns a base64-encoded JPEG string (much smaller than raw PNG).
    If the image is already within limits, it is still re-encoded as
    JPEG for consistent compression.
    """
    with Image.open(image_path) as img:
        width, height = img.size
        if max(width, height) > max_dim:
            ratio = max_dim / max(width, height)
            new_size = (int(width * ratio), int(height * ratio))
            img = img.resize(new_size, Image.LANCZOS)
            logger.info(
                "Downscaled image from %dx%d → %dx%d",
                width, height, new_size[0], new_size[1],
            )

        # Convert to RGB (in case of RGBA PNGs) and encode as JPEG
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=80)
        return base64.b64encode(buf.getvalue()).decode("utf-8")


def call_llm(
    prompt: str,
    system_prompt: Optional[str] = None,
    image_path: Optional[str] = None,
    response_format: Optional[Type[T]] = None,
    model: Optional[str] = None,
) -> Any:
    """Call the LLM via LiteLLM.

    Handles text, vision (when *image_path* is provided), and
    structured outputs (when *response_format* is a Pydantic model).
    """
    selected_model = model or (
        settings.get_active_vision_model()
        if image_path
        else settings.get_active_text_model()
    )

    messages = _build_messages(prompt, system_prompt, image_path)

    try:
        kwargs: dict[str, Any] = {
            "model": selected_model,
            "messages": messages,
            "timeout": settings.LLM_TIMEOUT_SECONDS,
        }

        if response_format:
            kwargs["response_format"] = response_format

        logger.info(
            "Calling LLM (%s) — prompt: %d chars, timeout: %ds",
            selected_model, len(prompt), settings.LLM_TIMEOUT_SECONDS,
        )
        response = completion(**kwargs)

        if response_format:
            content_str = response.choices[0].message.content
            return response_format.model_validate_json(content_str)

        return response.choices[0].message.content
    except Exception:
        logger.exception("Error calling LLM")
        raise


# ── Private helpers ─────────────────────────────────────────────────

def _build_messages(
    prompt: str,
    system_prompt: Optional[str],
    image_path: Optional[str],
) -> list[dict[str, Any]]:
    """Assemble the message list for a LiteLLM completion call."""
    messages: list[dict[str, Any]] = []

    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    if image_path:
        messages.append(_build_vision_message(prompt, image_path))
    else:
        messages.append({"role": "user", "content": prompt})

    return messages


def _build_vision_message(
    prompt: str, image_path: str,
) -> dict[str, Any]:
    """Create a multimodal user message with text + downscaled image."""
    base64_image = _downscale_image(
        image_path, settings.MAX_IMAGE_DIMENSION,
    )
    # Always JPEG after downscale
    mime_type = "image/jpeg"

    content = [
        {"type": "text", "text": prompt},
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:{mime_type};base64,{base64_image}",
            },
        },
    ]
    return {"role": "user", "content": content}
