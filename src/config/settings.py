import os
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """Configuration settings for UI Analyser."""
    
    # Provider APIs
    # LiteLLM will use these to route to the correct models
    GEMINI_API_KEY: Optional[str] = None
    OPENAI_API_KEY: Optional[str] = None
    ANTHROPIC_API_KEY: Optional[str] = None
    
    # Model selections
    VISION_MODEL: str = "gemini/gemini-2.5-flash"
    TEXT_MODEL: str = "gemini/gemini-2.5-flash"
    IMAGE_GENERATION_MODEL: str = "gemini-2.5-flash-image" # Fallback if we decide to implement image generation API

    # UI Configuration
    PLAYWRIGHT_TIMEOUT_MS: int = 60000

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()
