"""Utility for resizing and compressing images to optimize vision LLM calls.

Provides downscaling and compression to JPEG format, reducing file size and network payloads.
"""

import os
import logging
from typing import Optional
from PIL import Image
from src.config.settings import settings

logger = logging.getLogger(__name__)


def minimize_image(image_path: str, max_dim: Optional[int] = None) -> str:
    """Resize and compress the image at image_path to minimize size.

    If it is a PNG or has transparency, it is converted to RGB (with a white background)
    and saved as a highly compressed JPEG. The original raw file is deleted to save space.
    """
    if not image_path or not os.path.exists(image_path):
        return image_path

    if max_dim is None:
        max_dim = settings.MAX_IMAGE_DIMENSION

    logger.info("Minimizing image size: %s (max_dim=%d)", image_path, max_dim)
    try:
        with Image.open(image_path) as img:
            width, height = img.size

            # Downscale if needed
            if max(width, height) > max_dim:
                ratio = max_dim / max(width, height)
                new_size = (int(width * ratio), int(height * ratio))
                
                # Support both modern Pillow versions (PIL >= 9.1.0) and older ones
                try:
                    resampling_method = Image.Resampling.LANCZOS
                except AttributeError:
                    resampling_method = Image.LANCZOS

                img = img.resize(new_size, resampling_method)
                logger.info(
                    "Resized image from %dx%d to %dx%d",
                    width, height, new_size[0], new_size[1],
                )

            # Convert RGBA/P to RGB (handling transparency by pasting on a white background)
            if img.mode in ("RGBA", "LA") or (
                img.mode == "P" and "transparency" in img.info
            ):
                background = Image.new("RGB", img.size, (255, 255, 255))
                if img.mode == "RGBA":
                    background.paste(img, mask=img.split()[-1])
                else:
                    rgba_img = img.convert("RGBA")
                    background.paste(rgba_img, mask=rgba_img.split()[-1])
                img = background
            elif img.mode != "RGB":
                img = img.convert("RGB")

            # Define output path (convert extension to .jpg for maximum compression)
            base, _ = os.path.splitext(image_path)
            output_path = f"{base}_minimized.jpg"

            # Save optimized JPEG
            img.save(output_path, format="JPEG", quality=85, optimize=True)
            logger.info(
                "Saved minimized image to %s (size: %d bytes)",
                output_path, os.path.getsize(output_path),
            )

            # Delete original if it's different from the output path
            if image_path != output_path and os.path.exists(image_path):
                try:
                    os.remove(image_path)
                    logger.info("Removed original image file: %s", image_path)
                except Exception as e:
                    logger.warning("Could not remove original image: %s", e)

            return output_path
    except Exception as e:
        logger.exception("Failed to minimize image %s: %s", image_path, e)
        return image_path
