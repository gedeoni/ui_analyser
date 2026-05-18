"""System environment checks for Ollama availability and model discovery."""

import logging
import subprocess

import requests

logger = logging.getLogger(__name__)


def is_ollama_installed() -> bool:
    """Check if the Ollama CLI is installed on the system."""
    try:
        subprocess.run(
            ["ollama", "--version"],
            capture_output=True,
            text=True,
            check=True,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def get_ollama_models(only_vision: bool = False) -> list[str]:
    """Retrieve list of locally available Ollama models.

    If *only_vision* is True, filters to models with vision capabilities.
    """
    try:
        response = requests.get(
            "http://localhost:11434/api/tags", timeout=2.0,
        )
        response.raise_for_status()

        models_data = response.json().get("models", [])
        all_models = [model["name"] for model in models_data]

        if not only_vision:
            return all_models

        return _filter_vision_models(all_models)
    except requests.exceptions.RequestException:
        logger.warning("Ollama API not available (service might be down)")
        return []


def _filter_vision_models(model_names: list[str]) -> list[str]:
    """Query Ollama API for each model and keep only vision-capable ones."""
    vision_models: list[str] = []
    for name in model_names:
        try:
            resp = requests.post(
                "http://localhost:11434/api/show",
                json={"name": name},
                timeout=2.0,
            )
            if resp.status_code == 200:
                capabilities = resp.json().get("capabilities", [])
                if capabilities and "vision" in capabilities:
                    vision_models.append(name)
        except requests.exceptions.RequestException:
            logger.warning(
                "Could not check vision capability for %s", name,
            )
    return vision_models


def has_vision_model(model_name: str = "llama3.2-vision:latest") -> bool:
    """Check if a specific vision-capable model is available locally."""
    return model_name in get_ollama_models(only_vision=True)
