import os
import logging
from datetime import datetime
from typing import Optional
from src.config.settings import settings

logger = logging.getLogger(__name__)

async def generate_improved_landing_page(prompt: str, reference_image_path: Optional[str] = None, output_dir: str = "artifacts") -> str:
    """
    Generates an improved landing page based on the detailed prompt.
    Abstracts the underlying image generation model (e.g. Gemini, DALL-E).
    Returns the file path to the saved generated image.
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"improved_design_{timestamp}.png"
    filepath = os.path.join(output_dir, filename)

    logger.info(f"Generating improved landing page with prompt length: {len(prompt)}")
    
    # We use Google GenAI natively for Gemini image models since they support image+text inputs for editing
    if "gemini" in settings.IMAGE_GENERATION_MODEL.lower():
        from google import genai
        from google.genai import types
        
        client = genai.Client(api_key=settings.GEMINI_API_KEY)
        
        content_parts = [types.Part.from_text(text=prompt)]
        
        if reference_image_path and os.path.exists(reference_image_path):
            with open(reference_image_path, "rb") as f:
                image_bytes = f.read()
            # Assuming PNG for simplicity based on our web_capture tool
            image_part = types.Part.from_bytes(data=image_bytes, mime_type="image/png")
            content_parts.append(image_part)
            
        contents = [
            types.Content(
                role="user",
                parts=content_parts,
            ),
        ]
        
        config = types.GenerateContentConfig(
            response_modalities=["IMAGE", "TEXT"],
        )
        
        # We use standard generation (not stream) for simplicity in this wrapper
        response = client.models.generate_content(
            model=settings.IMAGE_GENERATION_MODEL,
            contents=contents,
            config=config,
        )
        
        # Extract the image from the response
        if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                if part.inline_data and part.inline_data.data:
                    with open(filepath, "wb") as f:
                        f.write(part.inline_data.data)
                    logger.info(f"Successfully saved generated image to {filepath}")
                    return filepath
                    
        raise Exception("Model did not return an image.")
    else:
        # Fallback to LiteLLM for other providers like OpenAI DALL-E 3
        # Note: DALL-E 3 doesn't easily accept reference images in the same way, so it will just use the prompt.
        from litellm import image_generation
        
        response = image_generation(
            prompt=prompt,
            model=settings.IMAGE_GENERATION_MODEL
        )
        
        # Download the image from the URL returned by LiteLLM (e.g. OpenAI)
        import requests
        image_url = response.data[0].url
        img_data = requests.get(image_url).content
        with open(filepath, 'wb') as handler:
            handler.write(img_data)
            
        logger.info(f"Successfully saved generated image to {filepath}")
        return filepath
