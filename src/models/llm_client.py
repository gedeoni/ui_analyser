import logging
import base64
from typing import Type, TypeVar, Optional, Any
from pydantic import BaseModel
from litellm import completion
from src.config.settings import settings

logger = logging.getLogger(__name__)

T = TypeVar('T', bound=BaseModel)

def encode_image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def call_llm(
    prompt: str,
    system_prompt: Optional[str] = None,
    image_path: Optional[str] = None,
    response_format: Optional[Type[T]] = None,
    model: Optional[str] = None
) -> Any:
    """
    Calls the LLM using LiteLLM.
    Handles text, vision (if image_path provided), and structured outputs.
    """
    selected_model = model or (settings.get_active_vision_model())
    
    messages = []
    
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
        
    if image_path:
        base64_image = encode_image_to_base64(image_path)
        mime_type = "image/png" if image_path.endswith('.png') else "image/jpeg"
        
        content = [
            {"type": "text", "text": prompt},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{mime_type};base64,{base64_image}"
                }
            }
        ]
        messages.append({"role": "user", "content": content})
    else:
        messages.append({"role": "user", "content": prompt})

    try:
        kwargs = {
            "model": selected_model,
            "messages": messages,
        }
        
        if response_format:
            kwargs["response_format"] = response_format
            
        logger.info(f"Calling LLM ({selected_model}) with prompt length: {len(prompt)}")
        response = completion(**kwargs)
        
        if response_format:
            # Most models support structured output. Depending on the model (e.g. OpenAI vs Gemini), 
            # LiteLLM parses it and returns it in response.choices[0].message.content (as stringified JSON).
            # We parse it into the Pydantic model.
            content_str = response.choices[0].message.content
            return response_format.model_validate_json(content_str)
            
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Error calling LLM: {e}")
        raise e
