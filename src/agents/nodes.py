import logging
from typing import Dict, Any, cast

from src.state.schema import GraphState, AnalysisResult, DesignPlan
from src.prompts.templates import (
    INFO_AGENT_PROMPT,
    ROUTER_PROMPT,
    UI_CRITIC_PROMPT,
    DESIGN_STRATEGIST_PROMPT,
    VISUAL_IMPLEMENTER_PROMPT,
    DESIGN_EDITOR_PROMPT
)
from src.models.llm_client import call_llm
from src.tools.web_capture import capture_website_screenshot
from src.tools.image_generation import generate_improved_landing_page

logger = logging.getLogger(__name__)

async def route_request(state: GraphState) -> Dict[str, Any]:
    """Determines the next step based on the user's latest message."""
    last_message = state["messages"][-1]["content"]
    
    # If target_url or current_image_path was explicitly set prior, use it as context.
    prompt = f"User Request: {last_message}\n\n"
    if state.get("target_url"):
        prompt += f"Context: User provided URL {state['target_url']}\n"
        
    decision = call_llm(prompt=prompt, system_prompt=ROUTER_PROMPT)
    decision = decision.strip().strip("'").strip('"')
    
    # Clean up the output in case the LLM was verbose
    valid_routes = ["capture_website", "analysis_pipeline", "design_editor", "info_agent"]
    for route in valid_routes:
        if route in decision:
            logger.info(f"Router decided: {route}")
            return {"messages": [{"role": "system", "content": f"Routed to: {route}"}], "next_node": route}
            
    logger.warning(f"Router returned invalid choice: {decision}. Defaulting to info_agent.")
    return {"messages": [{"role": "system", "content": f"Routed to: info_agent"}], "next_node": "info_agent"}

async def info_agent_node(state: GraphState) -> Dict[str, Any]:
    """Handles general questions."""
    last_message = state["messages"][-1]["content"]
    response = call_llm(prompt=last_message, system_prompt=INFO_AGENT_PROMPT)
    return {"messages": [{"role": "assistant", "content": response}]}

async def capture_website_node(state: GraphState) -> Dict[str, Any]:
    """Captures a screenshot of the provided URL."""
    # Simple extraction of URL from the last message. In a real scenario, use regex or an LLM call.
    last_message = state["messages"][-1]["content"]
    
    # Quick LLM call to extract URL cleanly
    url = call_llm(prompt=f"Extract only the URL from this text: {last_message}. If no URL, return 'NONE'.")
    url = url.strip()
    
    if url == "NONE":
        return {"current_error": "Could not find a valid URL in the request.", "messages": [{"role": "assistant", "content": "I couldn't find a valid URL in your request. Please provide one."}]}
        
    try:
        if not url.startswith("http"):
            url = "https://" + url
            
        filepath = await capture_website_screenshot(url)
        return {
            "target_url": url,
            "current_image_path": filepath,
            "messages": [{"role": "assistant", "content": f"Successfully captured a screenshot of {url}. I am analyzing it now."}],
            # Automatically proceed to analysis_pipeline by setting a flag or via edge routing
        }
    except Exception as e:
        return {"current_error": str(e), "messages": [{"role": "assistant", "content": f"Error capturing website: {e}"}]}

async def ui_critic_node(state: GraphState) -> Dict[str, Any]:
    """Analyzes the image and provides feedback."""
    image_path = state.get("current_image_path")
    if not image_path:
        return {"current_error": "No image to analyze.", "messages": [{"role": "assistant", "content": "I don't have an image to analyze. Please provide a URL."}]}
        
    try:
        result = call_llm(
            prompt="Analyze this landing page design.",
            system_prompt=UI_CRITIC_PROMPT,
            image_path=image_path,
            response_format=AnalysisResult
        )
        
        # Convert structured result to a string report for the next agent
        report = f"OVERALL IMPRESSION: {result.overall_impression}\n\nSTRENGTHS:\n" + "\n".join(f"- {s}" for s in result.strengths)
        report += f"\n\nCRITICAL ISSUES:\n" + "\n".join(f"- {i}" for i in result.critical_issues)
        report += f"\n\nPRIORITIES:\n" + "\n".join(f"- {p}" for p in result.top_3_priorities)
        
        # We store the string version for the prompt, but we could store the Pydantic object too.
        return {"analysis_report": report}
    except Exception as e:
        return {"current_error": str(e), "messages": [{"role": "assistant", "content": f"Error during analysis: {e}"}]}

async def design_strategist_node(state: GraphState) -> Dict[str, Any]:
    """Creates a design improvement plan based on the UI Critic's report."""
    report = state.get("analysis_report")
    if not report:
        return {"current_error": "No analysis report available."}
        
    try:
        result = call_llm(
            prompt="Create a design improvement plan.",
            system_prompt=DESIGN_STRATEGIST_PROMPT.format(analysis_report=report),
            response_format=DesignPlan
        )
        
        # Convert structured result to a string report
        plan = f"STRATEGY OVERVIEW: {result.strategy_overview}\n\nCOLORS:\n" + "\n".join(f"- {k}: {v}" for k,v in result.color_palette.items())
        plan += f"\n\nTYPOGRAPHY:\n" + "\n".join(f"- {k}: {v}" for k,v in result.typography.items())
        plan += f"\n\nLAYOUT IMPROVEMENTS:\n" + "\n".join(f"- {i}" for i in result.layout_improvements)
        plan += f"\n\nCTA OPTIMIZATION:\n" + "\n".join(f"- {i}" for i in result.cta_optimization)
        
        return {"design_plan": plan}
    except Exception as e:
        return {"current_error": str(e), "messages": [{"role": "assistant", "content": f"Error during strategy planning: {e}"}]}

async def visual_implementer_node(state: GraphState) -> Dict[str, Any]:
    """Generates the improved landing page image."""
    report = state.get("analysis_report")
    plan = state.get("design_plan")
    original_image = state.get("current_image_path")
    
    if not report or not plan:
        return {"current_error": "Missing analysis or design plan."}
        
    try:
        # Generate the prompt for image generation
        image_gen_prompt = call_llm(
            prompt="Generate the detailed image prompt.",
            system_prompt=VISUAL_IMPLEMENTER_PROMPT.format(analysis_report=report, design_plan=plan)
        )
        
        # Call the image generation tool
        filepath = await generate_improved_landing_page(prompt=image_gen_prompt, reference_image_path=original_image)
        
        response_msg = "✅ **Analysis & Generation Complete!**\n\n"
        response_msg += "I have analyzed your design, created an improvement strategy, and generated an updated version.\n\n"
        response_msg += f"**Saved new design to:** `{filepath}`\n\n"
        response_msg += f"**Key Priorities Addressed:**\n{report.split('PRIORITIES:')[1].strip()}\n"
        
        return {
            "messages": [{"role": "assistant", "content": response_msg}],
            "artifact_versions": {"latest_improved_design": 1},
            "artifact_filenames": {"latest_improved_design": filepath},
            "current_image_path": filepath # Update the current image to the new one for further edits
        }
    except Exception as e:
        return {"current_error": str(e), "messages": [{"role": "assistant", "content": f"Error generating image: {e}"}]}

async def design_editor_node(state: GraphState) -> Dict[str, Any]:
    """Edits an existing design based on user feedback."""
    last_message = state["messages"][-1]["content"]
    image_path = state.get("current_image_path")
    
    if not image_path:
        return {"messages": [{"role": "assistant", "content": "There is no current design to edit. Please analyze a page first."}]}
        
    try:
        # Generate the edit prompt
        edit_prompt = call_llm(
            prompt="Generate the edit prompt for the image generation model.",
            system_prompt=DESIGN_EDITOR_PROMPT.format(user_request=last_message)
        )
        
        # Call the image generation tool using the last generated image as reference
        filepath = await generate_improved_landing_page(prompt=edit_prompt, reference_image_path=image_path)
        
        # Update versions
        current_version = state.get("artifact_versions", {}).get("latest_improved_design", 1)
        new_version = current_version + 1
        
        response_msg = f"✅ **Design Edited Successfully!**\n\n"
        response_msg += f"**Saved updated design to:** `{filepath}` (Version {new_version})\n"
        
        return {
            "messages": [{"role": "assistant", "content": response_msg}],
            "artifact_versions": {"latest_improved_design": new_version},
            "artifact_filenames": {"latest_improved_design": filepath},
            "current_image_path": filepath
        }
    except Exception as e:
        return {"current_error": str(e), "messages": [{"role": "assistant", "content": f"Error editing design: {e}"}]}
