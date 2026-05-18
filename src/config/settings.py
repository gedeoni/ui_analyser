"""Application-wide configuration settings backed by pydantic-settings."""

from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuration settings for UI Analyser."""

    # Provider APIs
    # LiteLLM will use these to route to the correct models
    GEMINI_API_KEY: Optional[str] = None
    OPENAI_API_KEY: Optional[str] = None
    ANTHROPIC_API_KEY: Optional[str] = None

    # ── Model Provider Toggle ───────────────────────────
    # When True, use a cloud API (Gemini/OpenAI/Anthropic) instead of Ollama.
    USE_CLOUD: bool = False
    CLOUD_PROVIDER: str = "Gemini"  # "Gemini", "OpenAI", "Anthropic"

    # ── Model Selections ────────────────────────────────
    # Defaults target local Ollama models.  The app sidebar can override these
    # at runtime, or they can be set in .env / environment variables.
    VISION_MODEL: str = "ollama/llama3.2-vision:latest"
    TEXT_MODEL: str = "ollama/llama3.2:latest"
    IMAGE_GENERATION_MODEL: str = "gemini-2.5-flash-image"  # Cloud-only — no local equivalent

    # Cloud model defaults (used when USE_CLOUD is True)
    CLOUD_VISION_MODEL: str = "gemini/gemini-2.5-flash"
    CLOUD_TEXT_MODEL: str = "gemini/gemini-2.5-flash"

    # UI Configuration
    PLAYWRIGHT_TIMEOUT_MS: int = 60000

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Convenience helpers ─────────────────────────────
    def get_active_vision_model(self) -> str:
        """Return the vision model string to pass to LiteLLM."""
        if self.USE_CLOUD:
            return self.CLOUD_VISION_MODEL
        return self.VISION_MODEL

    def get_active_text_model(self) -> str:
        """Return the text model string to pass to LiteLLM."""
        if self.USE_CLOUD:
            return self.CLOUD_TEXT_MODEL
        return self.TEXT_MODEL


settings = Settings()
