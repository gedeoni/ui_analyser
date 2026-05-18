import logging
import os
import re
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

def _get_last_user_message(state: GraphState) -> str:
    """Walk backwards through messages to find the last user message.
    
    The router node appends a system message (e.g. 'Routed to: capture_website')
    which becomes messages[-1]. Downstream nodes need the original user input,
    not the router's system message.
    """
    for msg in reversed(state["messages"]):
        if msg.get("role") == "user":
            return msg["content"]
    # Fallback: return the very last message content if no user message found
    return state["messages"][-1]["content"]

async def route_request(state: GraphState) -> Dict[str, Any]:
    """Determines the next step based on the user's latest message."""
    last_message = state["messages"][-1]["content"]
    logger.info("==================================================")
    logger.info("🚀 STARTING WORKFLOW ROUTER")
    logger.info(f"User Message: {repr(last_message)}")
    
    # If target_url or current_image_path was explicitly set prior, use it as context.
    prompt = f"User Request: {last_message}\n\n"
    if state.get("target_url"):
        prompt += f"Context: User provided URL {state['target_url']}\n"
        logger.info(f"Router Context URL: {state['target_url']}")
    if state.get("current_image_path"):
        logger.info(f"Router Context Image: {state['current_image_path']}")
        
    decision = call_llm(prompt=prompt, system_prompt=ROUTER_PROMPT)
    decision = decision.strip().strip("'").strip('"')
    
    # Clean up the output in case the LLM was verbose
    valid_routes = ["capture_website", "analysis_pipeline", "design_editor", "info_agent"]
    for route in valid_routes:
        if route in decision:
            logger.info(f"🎯 Router decision matched: '{route}' (raw decision: '{decision}')")
            logger.info("==================================================")
            return {"messages": [{"role": "system", "content": f"Routed to: {route}"}], "next_node": route}
            
    logger.warning(f"⚠️ Router returned invalid/unexpected choice: '{decision}'. Defaulting to info_agent.")
    logger.info("==================================================")
    return {"messages": [{"role": "system", "content": f"Routed to: info_agent"}], "next_node": "info_agent"}

async def info_agent_node(state: GraphState) -> Dict[str, Any]:
    """Handles general questions."""
    last_message = _get_last_user_message(state)
    logger.info("==================================================")
    logger.info("ℹ️ RUNNING INFO_AGENT NODE")
    logger.info(f"Prompting LLM with general query: {repr(last_message)}")
    response = call_llm(prompt=last_message, system_prompt=INFO_AGENT_PROMPT)
    logger.info("Finished running info_agent node successfully.")
    logger.info("==================================================")
    return {"messages": [{"role": "assistant", "content": response}]}

async def capture_website_node(state: GraphState) -> Dict[str, Any]:
    """Captures a screenshot of the provided URL."""
    last_message = _get_last_user_message(state)
    logger.info("==================================================")
    logger.info("📸 RUNNING CAPTURE_WEBSITE NODE")
    logger.info(f"Extracting URL from message: {repr(last_message)}")
    
    # Try regex extraction first for maximum speed and reliability
    url_pattern = re.compile(r'(?:https?://)?(?:[a-zA-Z0-9-]+\.)+[a-zA-Z]{2,}(?:/[^\s]*)?')
    match = url_pattern.search(last_message)
    if match:
        url = match.group(0).rstrip('.,;!?')
        logger.info(f"✅ Extracted URL using regex: '{url}'")
    else:
        # Quick LLM call to extract URL cleanly if regex doesn't match directly
        logger.info("Regex extraction failed. Falling back to LLM URL extraction...")
        llm_extracted = call_llm(prompt=f"Extract only the URL from this text: {last_message}. If no URL, return 'NONE'.")
        llm_extracted = llm_extracted.strip()
        match_llm = url_pattern.search(llm_extracted)
        if match_llm and "NONE" not in llm_extracted:
            url = match_llm.group(0).rstrip('.,;!?')
            logger.info(f"✅ Extracted URL from LLM output using regex: '{url}'")
        else:
            url = "NONE"
            logger.info("❌ No URL found in request by regex or LLM.")
    
    if url == "NONE":
        logger.warning("No valid URL could be determined. Aborting capture website node.")
        logger.info("==================================================")
        return {"current_error": "Could not find a valid URL in the request.", "messages": [{"role": "assistant", "content": "I couldn't find a valid URL in your request. Please provide one."}]}
        
    try:
        if not url.startswith("http"):
            url = "https://" + url
            logger.info(f"Prepend 'https://' to URL: '{url}'")
            
        logger.info(f"Initiating Playwright to capture screenshot of URL: '{url}'")
        filepath = await capture_website_screenshot(url)
        logger.info(f"✅ Playwright captured website successfully: '{filepath}'")
        logger.info("==================================================")
        return {
            "target_url": url,
            "current_image_path": filepath,
            "messages": [{"role": "assistant", "content": f"Successfully captured a screenshot of {url}. I am analyzing it now."}],
        }
    except Exception as e:
        logger.error(f"💥 Error capturing website screenshot: {e}")
        logger.info("==================================================")
        return {"current_error": str(e), "messages": [{"role": "assistant", "content": f"Error capturing website: {e}"}]}

async def ui_critic_node(state: GraphState) -> Dict[str, Any]:
    """Analyzes the image and provides feedback."""
    image_path = state.get("current_image_path")
    logger.info("==================================================")
    logger.info("🔍 RUNNING UI_CRITIC NODE")
    logger.info(f"Target image path for analysis: '{image_path}'")
    if not image_path:
        logger.error("❌ No image path found in state!")
        logger.info("==================================================")
        return {"current_error": "No image to analyze.", "messages": [{"role": "assistant", "content": "I don't have an image to analyze. Please provide a URL."}]}
        
    try:
        logger.info("Calling multimodal LLM for visual landing page critique...")
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
        
        logger.info(f"✅ UI Critic successfully analyzed image.")
        logger.info(f"- Overall impression summary: '{result.overall_impression[:80]}...'")
        logger.info(f"- Strengths identified: {len(result.strengths)}")
        logger.info(f"- Critical issues found: {len(result.critical_issues)}")
        logger.info("==================================================")
        return {"analysis_report": report}
    except Exception as e:
        logger.error(f"💥 Error during visual analysis in UI Critic: {e}")
        logger.info("==================================================")
        return {"current_error": str(e), "messages": [{"role": "assistant", "content": f"Error during analysis: {e}"}]}

async def design_strategist_node(state: GraphState) -> Dict[str, Any]:
    """Creates a design improvement plan based on the UI Critic's report."""
    report = state.get("analysis_report")
    logger.info("==================================================")
    logger.info("🧠 RUNNING DESIGN_STRATEGIST NODE")
    if not report:
        logger.error("❌ No analysis report found in state!")
        logger.info("==================================================")
        return {"current_error": "No analysis report available."}
        
    try:
        logger.info("Calling LLM to formulate custom design strategy plan...")
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
        
        logger.info(f"✅ Design Strategist strategy plan generated successfully.")
        logger.info(f"- Strategy overview: '{result.strategy_overview[:80]}...'")
        logger.info(f"- Layout improvements planned: {len(result.layout_improvements)}")
        logger.info(f"- CTA optimizations planned: {len(result.cta_optimization)}")
        logger.info("==================================================")
        
        # Build a user-facing summary combining analysis + strategy
        # This is shown if the pipeline stops here (no image gen API key)
        has_image_key = any([
            os.getenv("GEMINI_API_KEY"),
            os.getenv("OPENAI_API_KEY"),
        ])
        
        response_parts = [
            "## 🔍 UI/UX Analysis Report\n",
            report,
            "\n---\n## 🧠 Design Improvement Strategy\n",
            plan,
        ]
        
        if not has_image_key:
            response_parts.append(
                "\n---\n> ⚠️ **Image generation skipped** — no API key configured. "
                "Add a Gemini or OpenAI API key in the sidebar to enable visual redesign generation."
            )
        
        return {
            "design_plan": plan,
            "messages": [{"role": "assistant", "content": "\n".join(response_parts)}]
        }
    except Exception as e:
        logger.error(f"💥 Error during design strategy formulation: {e}")
        logger.info("==================================================")
        return {"current_error": str(e), "messages": [{"role": "assistant", "content": f"Error during strategy planning: {e}"}]}

async def visual_implementer_node(state: GraphState) -> Dict[str, Any]:
    """Generates the improved landing page image."""
    report = state.get("analysis_report")
    plan = state.get("design_plan")
    original_image = state.get("current_image_path")
    logger.info("==================================================")
    logger.info("🎨 RUNNING VISUAL_IMPLEMENTER NODE")
    logger.info(f"Original reference image path: '{original_image}'")
    
    if not report or not plan:
        logger.error("❌ Missing analysis_report or design_plan in state!")
        logger.info("==================================================")
        return {"current_error": "Missing analysis or design plan."}
        
    try:
        logger.info("Calling LLM to synthesize visual critique + design plan into a detailed image generation prompt...")
        image_gen_prompt = call_llm(
            prompt="Generate the detailed image prompt.",
            system_prompt=VISUAL_IMPLEMENTER_PROMPT.format(analysis_report=report, design_plan=plan)
        )
        logger.info(f"Synthesized visual prompt length: {len(image_gen_prompt)} chars")
        logger.info("Triggering image generation model (calling generate_improved_landing_page)...")
        
        filepath = await generate_improved_landing_page(prompt=image_gen_prompt, reference_image_path=original_image)
        logger.info(f"✅ Improved design image generated successfully at: '{filepath}'")
        
        response_msg = "✅ **Analysis & Generation Complete!**\n\n"
        response_msg += "I have analyzed your design, created an improvement strategy, and generated an updated version.\n\n"
        response_msg += f"**Saved new design to:** `{filepath}`\n\n"
        response_msg += f"**Key Priorities Addressed:**\n{report.split('PRIORITIES:')[1].strip()}\n"
        
        logger.info("==================================================")
        return {
            "messages": [{"role": "assistant", "content": response_msg}],
            "artifact_versions": {"latest_improved_design": 1},
            "artifact_filenames": {"latest_improved_design": filepath},
            "current_image_path": filepath
        }
    except Exception as e:
        logger.error(f"💥 Error during visual landing page generation: {e}")
        logger.info("==================================================")
        return {"current_error": str(e), "messages": [{"role": "assistant", "content": f"Error generating image: {e}"}]}

async def design_editor_node(state: GraphState) -> Dict[str, Any]:
    """Edits an existing design based on user feedback."""
    last_message = _get_last_user_message(state)
    image_path = state.get("current_image_path")
    logger.info("==================================================")
    logger.info("✏️ RUNNING DESIGN_EDITOR NODE")
    logger.info(f"User edit request: {repr(last_message)}")
    logger.info(f"Current reference image path: '{image_path}'")
    
    if not image_path:
        logger.warning("⚠️ No current design found to edit!")
        logger.info("==================================================")
        return {"messages": [{"role": "assistant", "content": "There is no current design to edit. Please analyze a page first."}]}
        
    try:
        logger.info("Calling LLM to generate edit prompt for image generation model based on user request...")
        edit_prompt = call_llm(
            prompt="Generate the edit prompt for the image generation model.",
            system_prompt=DESIGN_EDITOR_PROMPT.format(user_request=last_message)
        )
        logger.info(f"Generated edit prompt: '{edit_prompt}'")
        logger.info("Triggering image editing generation...")
        
        filepath = await generate_improved_landing_page(prompt=edit_prompt, reference_image_path=image_path)
        logger.info(f"✅ Design successfully edited and saved at: '{filepath}'")
        
        current_version = state.get("artifact_versions", {}).get("latest_improved_design", 1)
        new_version = current_version + 1
        logger.info(f"Incrementing design version: {current_version} -> {new_version}")
        
        response_msg = f"✅ **Design Edited Successfully!**\n\n"
        response_msg += f"**Saved updated design to:** `{filepath}` (Version {new_version})\n"
        
        logger.info("==================================================")
        return {
            "messages": [{"role": "assistant", "content": response_msg}],
            "artifact_versions": {"latest_improved_design": new_version},
            "artifact_filenames": {"latest_improved_design": filepath},
            "current_image_path": filepath
        }
    except Exception as e:
        logger.error(f"💥 Error during design edit: {e}")
        logger.info("==================================================")
        return {"current_error": str(e), "messages": [{"role": "assistant", "content": f"Error editing design: {e}"}]}
