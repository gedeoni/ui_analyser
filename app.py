import streamlit as st
import os
import uuid
import asyncio
import re
from src.workflows.graph import app as agent_graph
from src.state.schema import GraphState
from src.config.settings import settings
from src.utils.system_checks import is_ollama_installed, get_ollama_models

st.set_page_config(page_title="UI Analyser Agent", page_icon="🎨", layout="wide")

st.title("🎨 UI/UX Feedback Agent Team")

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

# ── Model Selection Sidebar ────────────────────────────────
def render_model_settings():
    """Renders model selection UI in the sidebar — local Ollama first, cloud fallback."""
    st.sidebar.header("📦 Model Selection")

    ollama_installed = is_ollama_installed()

    if not ollama_installed:
        st.sidebar.warning("⚠️ Ollama is not installed. Falling back to Cloud API.")
        vision_models = []
    else:
        vision_models = get_ollama_models(only_vision=True)
        if not vision_models:
            st.sidebar.warning(
                "⚠️ No model with visual capabilities found locally.\n\n"
                "This agent requires a **vision model** to analyse screenshots.  "
                "Pull one with:\n```\nollama pull llama3.2-vision:latest\n```\n"
                "Or enable **Cloud API** below to use Gemini instead."
            )

    # Default cloud toggle state
    if "use_cloud" not in st.session_state:
        st.session_state.use_cloud = not bool(vision_models)

    if vision_models:
        st.session_state.use_cloud = st.sidebar.checkbox(
            "Use Cloud API instead of Ollama",
            value=st.session_state.use_cloud,
        )

    # ── Cloud mode ──────────────────────────────────────────
    if not vision_models or st.session_state.use_cloud:
        st.session_state.use_cloud = True
        settings.USE_CLOUD = True

        provider = st.sidebar.selectbox(
            "Select Cloud Provider", options=["Gemini", "OpenAI", "Anthropic"]
        )
        settings.CLOUD_PROVIDER = provider

        api_key = st.sidebar.text_input(f"{provider} API Key", type="password")
        if api_key:
            if provider == "Gemini":
                os.environ["GEMINI_API_KEY"] = api_key
                settings.GEMINI_API_KEY = api_key
            elif provider == "OpenAI":
                os.environ["OPENAI_API_KEY"] = api_key
                settings.OPENAI_API_KEY = api_key
            elif provider == "Anthropic":
                os.environ["ANTHROPIC_API_KEY"] = api_key
                settings.ANTHROPIC_API_KEY = api_key

        if provider == "Gemini":
            model = st.sidebar.selectbox(
                "Select Model",
                options=["gemini/gemini-2.5-flash", "gemini/gemini-2.5-pro"],
            )
        elif provider == "OpenAI":
            model = st.sidebar.selectbox(
                "Select Model", options=["openai/gpt-4o", "openai/gpt-4o-mini"]
            )
        elif provider == "Anthropic":
            model = st.sidebar.selectbox(
                "Select Model",
                options=["anthropic/claude-sonnet-4-20250514"],
            )

        settings.CLOUD_VISION_MODEL = model
        settings.CLOUD_TEXT_MODEL = model
        st.sidebar.info(f"Using **{model}** via {provider} cloud.")

        if not api_key:
            st.sidebar.info("Please enter your API key to proceed.")

    # ── Local Ollama mode ───────────────────────────────────
    else:
        settings.USE_CLOUD = False
        selected = st.sidebar.selectbox(
            "Select Local Vision Model",
            options=vision_models,
            index=vision_models.index("llama3.2-vision:latest")
            if "llama3.2-vision:latest" in vision_models
            else 0,
            help="Only models with vision capabilities are shown.",
        )
        settings.VISION_MODEL = f"ollama/{selected}"
        settings.TEXT_MODEL = f"ollama/{selected}"
        st.sidebar.info(f"Using **{selected}** locally via Ollama.")


# ── Render Sidebar ──────────────────────────────────────────
with st.sidebar:
    render_model_settings()

    st.markdown("---")
    st.markdown("### How it works")
    st.markdown("1. Provide a website URL.")
    st.markdown("2. Our AI agents will analyze the layout, typography, and UX.")
    st.markdown("3. A design strategist will create an improvement plan.")
    st.markdown("4. A visual implementer will generate a new design.")

# Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if "image" in msg:
            st.image(msg["image"])

# Input methods
def extract_url(text):
    url_pattern = re.compile(r'(?:https?://)?(?:[a-zA-Z0-9-]+\.)+[a-zA-Z]{2,}(?:/[^\s]*)?')
    match = url_pattern.search(text)
    if match:
        url = match.group(0)
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        return url
    return None

user_input = st.chat_input("Provide a website URL to analyse")

async def run_agent(state_input, config):
    # We use stream to get step-by-step updates
    async for event in agent_graph.astream(state_input, config=config):
        for node, state in event.items():
            if "messages" in state and len(state["messages"]) > 0:
                last_msg = state["messages"][-1]
                if last_msg["role"] == "assistant":
                    with st.chat_message("assistant"):
                        st.write(last_msg["content"])
                        st.session_state.messages.append({"role": "assistant", "content": last_msg["content"]})
                        
            # If a new image was generated, display it
            if "current_image_path" in state and state["current_image_path"]:
                img_path = state["current_image_path"]
                if os.path.exists(img_path):
                    st.image(img_path, caption="Current Design")
                    st.session_state.messages.append({"role": "assistant", "content": "Updated design.", "image": img_path})

if user_input:
    # Check that we have a valid configuration before proceeding
    if settings.USE_CLOUD and not any([
        os.getenv("GEMINI_API_KEY"),
        os.getenv("OPENAI_API_KEY"),
        os.getenv("ANTHROPIC_API_KEY"),
    ]):
        st.error("Please provide an API key in the sidebar.")
    else:
        # Construct initial state
        initial_state = {"messages": []}
        
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)
        initial_state["messages"].append({"role": "user", "content": user_input})

        config = {"configurable": {"thread_id": st.session_state.thread_id}}
        
        with st.spinner("Agents are thinking..."):
            asyncio.run(run_agent(initial_state, config))
