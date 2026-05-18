import subprocess
import requests
import logging

logger = logging.getLogger(__name__)


def is_ollama_installed() -> bool:
    """Check if the Ollama CLI is installed on the system."""
    try:
        subprocess.run(
            ["ollama", "--version"],
            capture_output=True,
            text=True,
            check=True
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def get_ollama_models(only_vision: bool = False) -> list[str]:
    """Retrieve list of locally available Ollama models.

    If only_vision is True, filters to show only models with vision capabilities.
    """
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=2.0)
        response.raise_for_status()

        models_data = response.json().get("models", [])
        all_models = [model["name"] for model in models_data]

        if not only_vision:
            return all_models

        # Filter for vision capability
        vision_models = []
        for model_name in all_models:
            try:
                show_resp = requests.post(
                    "http://localhost:11434/api/show",
                    json={"name": model_name},
                    timeout=2.0,
                )
                if show_resp.status_code == 200:
                    capabilities = show_resp.json().get("capabilities", [])
                    if capabilities and "vision" in capabilities:
                        vision_models.append(model_name)
            except Exception as e:
                logger.warning(f"Could not check vision for {model_name}: {e}")

        return vision_models
    except requests.exceptions.RequestException as e:
        logger.warning(f"Ollama API not available (service might be down): {e}")
        return []


def has_vision_model(model_name: str = "llama3.2-vision:latest") -> bool:
    """Check if a specific vision-capable model is available locally."""
    vision_models = get_ollama_models(only_vision=True)
    return model_name in vision_models
