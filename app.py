"""Streamlit UI for the UI/UX Feedback Agent Team.

Provides model selection (local Ollama or cloud), chat interface,
and async streaming of the LangGraph agent workflow.
"""

import asyncio
import logging
import os
import re
import uuid

import streamlit as st

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from src.config.settings import settings
from src.utils.system_checks import get_ollama_models, is_ollama_installed
from src.workflows.graph import workflow as agent_workflow

# ── Logging ─────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Page config ─────────────────────────────────────────────────────

st.set_page_config(
    page_title="UI Analyser Agent", page_icon="🎨", layout="wide",
)
st.title("🎨 UI/UX Feedback Agent Team")

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

# ── Provider ↔ env-var / model mapping ──────────────────────────────

_PROVIDER_ENV_KEYS: dict[str, str] = {
    "Gemini": "GEMINI_API_KEY",
    "OpenAI": "OPENAI_API_KEY",
    "Anthropic": "ANTHROPIC_API_KEY",
}

_PROVIDER_MODELS: dict[str, list[str]] = {
    "Gemini": ["gemini/gemini-2.5-flash", "gemini/gemini-2.5-pro"],
    "OpenAI": ["openai/gpt-4o", "openai/gpt-4o-mini"],
    "Anthropic": ["anthropic/claude-sonnet-4-20250514"],
}


# ── Model-selection sidebar helpers ─────────────────────────────────

def _apply_api_key(provider: str, api_key: str) -> None:
    """Store *api_key* in the environment and in ``settings``."""
    env_key = _PROVIDER_ENV_KEYS[provider]
    os.environ[env_key] = api_key
    setattr(settings, env_key, api_key)


def _render_cloud_settings() -> None:
    """Render cloud-provider selection, API key input, and model picker."""
    settings.USE_CLOUD = True
    st.session_state.use_cloud = True

    provider = st.sidebar.selectbox(
        "Select Cloud Provider", options=list(_PROVIDER_MODELS),
    )
    settings.CLOUD_PROVIDER = provider

    api_key = st.sidebar.text_input(f"{provider} API Key", type="password")
    if api_key:
        _apply_api_key(provider, api_key)

    model = st.sidebar.selectbox(
        "Select Model", options=_PROVIDER_MODELS[provider],
    )
    settings.CLOUD_VISION_MODEL = model
    settings.CLOUD_TEXT_MODEL = model
    st.sidebar.info(f"Using **{model}** via {provider} cloud.")

    if not api_key:
        st.sidebar.info("Please enter your API key to proceed.")


def _render_local_settings(vision_models: list[str]) -> None:
    """Render local Ollama model selectors for vision and text."""
    settings.USE_CLOUD = False

    default_vision = "llama3.2-vision:latest"
    vision_idx = (
        vision_models.index(default_vision)
        if default_vision in vision_models
        else 0
    )
    selected_vision = st.sidebar.selectbox(
        "Select Local Vision Model",
        options=vision_models,
        index=vision_idx,
        help="Only models with vision capabilities are shown.",
    )
    settings.VISION_MODEL = f"ollama/{selected_vision}"

    # Text model — all local models, defaulting to a lighter one
    all_local = get_ollama_models(only_vision=False)
    if all_local:
        default_text = "llama3.2:latest"
        text_idx = (
            all_local.index(default_text)
            if default_text in all_local
            else 0
        )
        selected_text = st.sidebar.selectbox(
            "Select Local Text Model",
            options=all_local,
            index=text_idx,
            help="Faster text-only model for routing and prompts.",
        )
        settings.TEXT_MODEL = f"ollama/{selected_text}"
    else:
        settings.TEXT_MODEL = f"ollama/{selected_vision}"

    st.sidebar.info(
        f"🔭 Vision: **{selected_vision}**\n\n"
        f"💬 Text: **{settings.TEXT_MODEL.removeprefix('ollama/')}**",
    )


def render_model_settings() -> None:
    """Render the model-selection UI in the sidebar.

    Prefers local Ollama vision models; falls back to cloud providers.
    """
    st.sidebar.header("📦 Model Selection")

    ollama_installed = is_ollama_installed()
    if not ollama_installed:
        st.sidebar.warning(
            "⚠️ Ollama is not installed. Falling back to Cloud API.",
        )
        vision_models: list[str] = []
    else:
        vision_models = get_ollama_models(only_vision=True)
        if not vision_models:
            st.sidebar.warning(
                "⚠️ No model with visual capabilities found locally.\n\n"
                "This agent requires a **vision model** to analyse "
                "screenshots.  Pull one with:\n"
                "```\nollama pull llama3.2-vision:latest\n```\n"
                "Or enable **Cloud API** below to use Gemini instead.",
            )

    # Default cloud toggle
    if "use_cloud" not in st.session_state:
        st.session_state.use_cloud = not bool(vision_models)

    if vision_models:
        st.session_state.use_cloud = st.sidebar.checkbox(
            "Use Cloud API instead of Ollama",
            value=st.session_state.use_cloud,
        )

    # Delegate to the appropriate sub-renderer
    if not vision_models or st.session_state.use_cloud:
        _render_cloud_settings()
    else:
        _render_local_settings(vision_models)


# ── Sidebar ─────────────────────────────────────────────────────────

with st.sidebar:
    render_model_settings()

    st.markdown("---")
    st.markdown("### How it works")
    st.markdown("1. Provide a website URL.")
    st.markdown(
        "2. Our AI agents will analyze the layout, typography, and UX.",
    )
    st.markdown("3. A design strategist will create an improvement plan.")
    st.markdown("4. A visual implementer will generate a new design.")


# ── Chat interface ──────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if "image" in msg:
            st.image(msg["image"])


# ── URL extraction helper ──────────────────────────────────────────

_URL_RE = re.compile(
    r"(?:https?://)?(?:[a-zA-Z0-9-]+\.)+[a-zA-Z]{2,}(?:/[^\s]*)?",
)


def extract_url(text: str) -> str | None:
    """Return the first URL found in *text*, or ``None``."""
    match = _URL_RE.search(text)
    if not match:
        return None
    url = match.group(0)
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    return url


# ── Agent runner ────────────────────────────────────────────────────

user_input = st.chat_input("Provide a website URL to analyse")


async def run_agent(
    state_input: dict, run_config: dict,
) -> None:
    """Stream agent workflow events and render them in the chat."""
    async with AsyncSqliteSaver.from_conn_string(
        "data/checkpoints.sqlite",
    ) as memory:
        agent_graph = agent_workflow.compile(checkpointer=memory)

        async for event in agent_graph.astream(
            state_input, config=run_config,
        ):
            _handle_stream_event(event)


def _handle_stream_event(event: dict) -> None:
    """Process a single streamed event from the agent graph."""
    for _node, state in event.items():
        _render_assistant_messages(state)
        _render_generated_image(state)


def _render_assistant_messages(state: dict) -> None:
    """Display any new assistant messages from the state update."""
    messages = state.get("messages", [])
    if not messages:
        return
    last_msg = messages[-1]
    if last_msg.get("role") != "assistant":
        return
    with st.chat_message("assistant"):
        st.write(last_msg["content"])
        st.session_state.messages.append(
            {"role": "assistant", "content": last_msg["content"]},
        )


def _render_generated_image(state: dict) -> None:
    """Show a newly generated image, if present in the state update."""
    img_path = state.get("current_image_path")
    if not img_path or not os.path.exists(img_path):
        return
    st.image(img_path, caption="Current Design")
    st.session_state.messages.append(
        {"role": "assistant", "content": "Updated design.", "image": img_path},
    )


# ── Main execution ──────────────────────────────────────────────────

if user_input:
    has_api_key = any((
        os.getenv("GEMINI_API_KEY"),
        os.getenv("OPENAI_API_KEY"),
        os.getenv("ANTHROPIC_API_KEY"),
    ))

    if settings.USE_CLOUD and not has_api_key:
        st.error("Please provide an API key in the sidebar.")
    else:
        initial_state: dict = {"messages": []}

        st.session_state.messages.append(
            {"role": "user", "content": user_input},
        )
        with st.chat_message("user"):
            st.write(user_input)
        initial_state["messages"].append(
            {"role": "user", "content": user_input},
        )

        config = {
            "configurable": {"thread_id": st.session_state.thread_id},
        }

        with st.spinner("Agents are thinking..."):
            asyncio.run(run_agent(initial_state, config))
