# UI/UX Feedback Agent Team (UI Analyser)

A sophisticated multi-agent system built with **LangGraph**, **LiteLLM**, and **Pydantic** that analyzes landing page designs, provides expert UI/UX feedback, and optionally generates improved versions using advanced multimodal models.

## Features

- **🌐 Website URL Support**: Simply provide a website URL, and the agent uses Playwright to capture a full-page screenshot for analysis.
- **👁️ Visual AI Analysis**: Agents automatically analyze layout, typography, colors, and UX patterns from screenshots.
- **🎯 Expert Feedback**: Comprehensive critique covering visual hierarchy, accessibility, conversion optimization, and design best practices.
- **✨ Automatic Improvements**: Generates improved landing page designs incorporating all recommendations (requires cloud API key).
- **🤖 Multi-Agent Orchestration**: Built on LangGraph with explicit routing, state management, and clear multi-agent pipelines.
- **🔌 Hybrid Model Support**: Run entirely locally with Ollama, or use cloud APIs (Gemini, OpenAI, Anthropic). The system intelligently splits workloads between a fast text model and a vision model.
- **💾 Memory Persistence**: Uses SQLite checkpointing for maintaining conversation and design history across sessions.
- **📊 Comprehensive Logging**: Every node logs its actions with clear visual markers for easy debugging and monitoring.

## Architecture

```mermaid
graph TD
    User([User Input]) --> Router
    Router --> |General Query| InfoAgent
    Router --> |Website URL| WebCapture
    Router --> |New Image / Analysis| UICritic
    Router --> |Edit Request| DesignEditor
    
    WebCapture --> |Screenshot captured| UICritic
    
    subgraph Analysis Pipeline
        UICritic --> |Analysis Report| DesignStrategist
        DesignStrategist --> |API Key?| Check{Has Image Gen Key?}
        Check --> |Yes| VisualImplementer
        Check --> |No| EndReport[Return Text Report]
    end
    
    VisualImplementer --> |New Image| User
    EndReport --> User
    DesignEditor --> |Edited Image| User
    InfoAgent --> User
```

The system uses a **StateGraph (Supervisor/Worker pattern)** with specialized agents:

1. **Router Agent (Supervisor)**: Determines if the request requires URL capturing, image analysis, design editing, or is a general query. Uses the fast text model.
2. **Info Agent**: Handles general Q&A about UI/UX design. Uses the text model.
3. **Capture Website Node**: Uses Playwright to screenshot provided URLs. URL extraction uses regex with an LLM fallback.
4. **Analysis Pipeline (Sequential Subgraph)**:
   - **UI Critic**: Analyzes design layout, hierarchy, and UX. Uses the **vision model** (the only node that requires it).
   - **Design Strategist**: Creates a detailed, actionable improvement plan with structured output. Uses the text model.
   - **Visual Implementer** *(conditional)*: Generates the improved landing page design. Requires a cloud API key (Gemini/OpenAI). If no key is configured, the pipeline gracefully stops at the strategist and returns the text-based report.
5. **Design Editor**: Edits existing generated designs based on iterative feedback.

### Model Selection Strategy

The system uses **two separate models** to optimize speed and resource usage:

| Model Role | Default (Local) | Used By |
|---|---|---|
| **Text Model** | `ollama/llama3.2:latest` (~2 GB) | Router, Info Agent, URL extraction, Strategist, Implementer prompt synthesis |
| **Vision Model** | `ollama/llama3.2-vision:latest` (~7.8 GB) | UI Critic (screenshot analysis) |

The `call_llm()` function automatically selects the right model: **vision model** when an image is provided, **text model** otherwise. This avoids loading the heavy vision model for simple text tasks, dramatically improving response times.

## Quick Start

### 1. Prerequisites
- Python 3.12+
- [Ollama](https://ollama.ai/) installed and running (for local mode)

### 2. Setup
```bash
# Clone or navigate to the directory
cd ui_analyser

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Install Playwright browsers (for website capturing)
playwright install chromium
```

### 3. Pull Ollama Models
```bash
# Required: Vision model for screenshot analysis
ollama pull llama3.2-vision:latest

# Recommended: Lighter text model for faster routing & strategy
ollama pull llama3.2:latest
```

### 4. Configuration (Optional)
For image generation and cloud model access, create a `.env` file:
```bash
cp .env.example .env
```
Add your API keys:
```env
GEMINI_API_KEY="your_gemini_api_key_here"
# Optional:
# OPENAI_API_KEY="your_openai_key"
# ANTHROPIC_API_KEY="your_anthropic_key"
```

> **Note**: Without a cloud API key, the pipeline will still run the full analysis and strategy — only the visual redesign generation step is skipped.

### 5. Run the Streamlit Interface
```bash
streamlit run app.py
```

## Project Structure

```
ui_analyser/
├── app.py                          # Streamlit UI entry point
├── requirements.txt                # Python dependencies
├── .env                            # API keys (not committed)
├── data/                           # SQLite checkpoints (auto-created)
├── artifacts/                      # Generated screenshots & designs
└── src/
    ├── agents/
    │   └── nodes.py                # All LangGraph node functions
    ├── config/
    │   └── settings.py             # Pydantic settings (models, keys, toggles)
    ├── models/
    │   └── llm_client.py           # Unified LLM caller (auto text/vision routing)
    ├── prompts/
    │   └── templates.py            # System prompts for each agent
    ├── state/
    │   └── schema.py               # GraphState + Pydantic structured outputs
    ├── tools/
    │   ├── web_capture.py          # Playwright screenshot tool
    │   └── image_generation.py     # Image generation tool (cloud API)
    ├── utils/
    │   └── system_checks.py        # Ollama detection & model listing
    └── workflows/
        └── graph.py                # LangGraph workflow definition & edges
```

## Sidebar Controls

The Streamlit sidebar provides runtime model configuration:

- **Cloud vs Local toggle**: Switch between Ollama and cloud APIs
- **Vision Model selector**: Choose which vision-capable model analyzes screenshots
- **Text Model selector**: Choose a lighter model for routing, strategy, and prompt tasks
- **Cloud Provider & API Key**: Configure Gemini, OpenAI, or Anthropic when using cloud mode

## Extending the System

- **Prompts**: Edit `src/prompts/templates.py` to change agent behavior.
- **State**: Add new tracking variables in `src/state/schema.py`.
- **Workflows**: Modify `src/workflows/graph.py` to add new agents or change the routing logic.
- **Models**: Update `src/config/settings.py` to add new model providers or change defaults.
- **Tools**: Add new tools in `src/tools/` and wire them into node functions in `src/agents/nodes.py`.
