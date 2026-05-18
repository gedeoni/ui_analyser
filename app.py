import streamlit as st
import os
import uuid
import asyncio
import re
from src.workflows.graph import app as agent_graph
from src.state.schema import GraphState
from src.config.settings import settings

st.set_page_config(page_title="UI Analyser Agent", page_icon="🎨", layout="wide")

st.title("🎨 UI/UX Feedback Agent Team")

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

# Initialize environment variables from UI if not set
with st.sidebar:
    st.header("Settings")
    gemini_key = st.text_input("Gemini API Key", type="password", value=os.getenv("GEMINI_API_KEY", ""))
    if gemini_key:
        os.environ["GEMINI_API_KEY"] = gemini_key
    
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
    if not os.getenv("GEMINI_API_KEY") and not os.getenv("OPENAI_API_KEY"):
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
