"""LangGraph agent node functions for the UI Analyser workflow.

Each public async function is a graph node that receives the shared
``GraphState`` and returns a partial state update dict.
"""

import logging
import os
import re
from typing import Any, Dict, Optional

from src.models.llm_client import call_llm
from src.prompts.templates import (
    DESIGN_EDITOR_PROMPT,
    DESIGN_STRATEGIST_PROMPT,
    INFO_AGENT_PROMPT,
    ROUTER_PROMPT,
    UI_CRITIC_PROMPT,
    VISUAL_IMPLEMENTER_PROMPT,
)
from src.state.schema import AnalysisResult, DesignPlan, GraphState
from src.tools.image_generation import generate_improved_landing_page
from src.tools.web_capture import capture_website_screenshot
from src.utils.image_optimizer import minimize_image

logger = logging.getLogger(__name__)

# Compiled once — reused across URL extraction calls.
_URL_PATTERN = re.compile(
    r"(?:https?://)?(?:[a-zA-Z0-9-]+\.)+[a-zA-Z]{2,}(?:/[^\s]*)?",
)

_VALID_ROUTES = frozenset(
    ("capture_website", "analysis_pipeline", "design_editor", "info_agent"),
)

_SECTION_SEP = "=" * 50


# ── Shared helpers ──────────────────────────────────────────────────


def _log_section(title: str) -> None:
    """Emit a visually distinct section header to the log."""
    logger.info(_SECTION_SEP)
    logger.info(title)


def _get_last_user_message(state: GraphState) -> str:
    """Walk backwards through messages to find the last user message.

    The router node appends a system message (e.g. ``Routed to: …``)
    which becomes ``messages[-1]``.  Downstream nodes need the original
    user input, not the router's system message.
    """
    for msg in reversed(state["messages"]):
        if msg.get("role") == "user":
            return msg["content"]
    # Fallback: return the very last message content
    return state["messages"][-1]["content"]


def _extract_url(text: str) -> Optional[str]:
    """Try regex first, then fall back to an LLM call to extract a URL.

    Returns ``None`` when no URL can be found.
    """
    match = _URL_PATTERN.search(text)
    if match:
        url = match.group(0).rstrip(".,;!?")
        logger.info("Extracted URL via regex: %s", url)
        return url

    logger.info("Regex miss — falling back to LLM URL extraction …")
    llm_text = call_llm(
        prompt=(
            f"Extract only the URL from this text: {text}. "
            "If no URL, return 'NONE'."
        ),
    ).strip()

    match = _URL_PATTERN.search(llm_text)
    if match and "NONE" not in llm_text:
        url = match.group(0).rstrip(".,;!?")
        logger.info("Extracted URL via LLM: %s", url)
        return url

    logger.info("No URL found by regex or LLM.")
    return None


def _format_analysis_report(result: AnalysisResult) -> str:
    """Render an ``AnalysisResult`` into a human-readable text report."""
    strengths = "\n".join(f"- {s}" for s in result.strengths)
    issues = "\n".join(f"- {i}" for i in result.critical_issues)
    priorities = "\n".join(f"- {p}" for p in result.top_3_priorities)
    return (
        f"OVERALL IMPRESSION: {result.overall_impression}\n\n"
        f"STRENGTHS:\n{strengths}\n\n"
        f"CRITICAL ISSUES:\n{issues}\n\n"
        f"PRIORITIES:\n{priorities}"
    )


def _format_design_plan(result: DesignPlan) -> str:
    """Render a ``DesignPlan`` into a human-readable text plan."""
    colors = (
        f"- Primary CTA Background: {result.color_palette.primary_cta_bg}\n"
        f"- Primary CTA Text: {result.color_palette.primary_cta_text}\n"
        f"- Page Background: {result.color_palette.background}\n"
        f"- Main Text Color: {result.color_palette.text_color}\n"
        f"- Accent Color: {result.color_palette.accent}"
    )
    typo = (
        f"- Font Family: {result.typography.font_family}\n"
        f"- Base Font Size: {result.typography.base_font_size}\n"
        f"- Line Height Ratio: {result.typography.line_height_ratio}\n"
        f"- Letter Spacing: {result.typography.letter_spacing}"
    )
    layout = "\n".join(f"- {i}" for i in result.layout_improvements)
    cta = "\n".join(f"- {i}" for i in result.cta_optimization)
    return (
        f"STRATEGY OVERVIEW: {result.strategy_overview}\n\n"
        f"COLORS:\n{colors}\n\n"
        f"TYPOGRAPHY:\n{typo}\n\n"
        f"LAYOUT IMPROVEMENTS:\n{layout}\n\n"
        f"CTA OPTIMIZATION:\n{cta}"
    )


def _error_response(
    error: str, user_message: str,
) -> Dict[str, Any]:
    """Build a standard error state update with a user-facing message."""
    return {
        "current_error": error,
        "messages": [{"role": "assistant", "content": user_message}],
    }


# ── Public graph nodes ──────────────────────────────────────────────

async def route_request(state: GraphState) -> Dict[str, Any]:
    """Determine the next step based on the user's latest message."""
    last_message = state["messages"][-1]["content"]
    _log_section("🚀 STARTING WORKFLOW ROUTER")
    logger.info("User message: %r", last_message)

    prompt = f"User Request: {last_message}\n\n"
    if state.get("target_url"):
        prompt += f"Context: User provided URL {state['target_url']}\n"
        logger.info("Router context URL: %s", state["target_url"])
    if state.get("current_image_path"):
        prompt += f"Context: User uploaded/provided image path: {state['current_image_path']}\n"
        logger.info("Router context image: %s", state["current_image_path"])

    decision = call_llm(prompt=prompt, system_prompt=ROUTER_PROMPT)
    decision = decision.strip().strip("'\"")

    for route in _VALID_ROUTES:
        if route in decision:
            logger.info(
                "Router matched route '%s' (raw: '%s')", route, decision,
            )
            logger.info(_SECTION_SEP)
            return {
                "messages": [
                    {"role": "system", "content": f"Routed to: {route}"},
                ],
                "next_node": route,
            }

    logger.warning(
        "Unexpected router output '%s' — defaulting to info_agent", decision,
    )
    logger.info(_SECTION_SEP)
    return {
        "messages": [
            {"role": "system", "content": "Routed to: info_agent"},
        ],
        "next_node": "info_agent",
    }


async def info_agent_node(state: GraphState) -> Dict[str, Any]:
    """Handle general questions."""
    last_message = _get_last_user_message(state)
    _log_section("ℹ️  RUNNING INFO_AGENT NODE")
    logger.info("Query: %r", last_message)

    response = call_llm(prompt=last_message, system_prompt=INFO_AGENT_PROMPT)

    logger.info("info_agent node completed.")
    logger.info(_SECTION_SEP)
    return {"messages": [{"role": "assistant", "content": response}]}


async def capture_website_node(state: GraphState) -> Dict[str, Any]:
    """Capture a screenshot of the URL found in the user's message."""
    last_message = _get_last_user_message(state)
    _log_section("📸 RUNNING CAPTURE_WEBSITE NODE")
    logger.info("Extracting URL from: %r", last_message)

    url = _extract_url(last_message)
    if url is None:
        logger.warning("No valid URL found — aborting capture.")
        logger.info(_SECTION_SEP)
        return _error_response(
            "Could not find a valid URL in the request.",
            "I couldn't find a valid URL in your request. Please provide one.",
        )

    if not url.startswith("http"):
        url = "https://" + url
        logger.info("Prepended scheme: %s", url)

    try:
        logger.info("Capturing screenshot of %s …", url)
        filepath = await capture_website_screenshot(url)
        logger.info("Screenshot saved: %s. Minimizing size...", filepath)
        filepath = minimize_image(filepath)
        logger.info("Minimized screenshot saved to: %s", filepath)
        logger.info(_SECTION_SEP)
        return {
            "target_url": url,
            "current_image_path": filepath,
            "messages": [
                {
                    "role": "assistant",
                    "content": (
                        f"Successfully captured a screenshot of {url}. "
                        "I am analyzing it now."
                    ),
                },
            ],
        }
    except Exception as exc:
        logger.exception("Error capturing website screenshot")
        logger.info(_SECTION_SEP)
        return _error_response(
            str(exc), f"Error capturing website: {exc}",
        )


async def ui_critic_node(state: GraphState) -> Dict[str, Any]:
    """Analyze the captured image and produce a structured report."""
    image_path = state.get("current_image_path")
    _log_section("🔍 RUNNING UI_CRITIC NODE")
    logger.info("Image path: %s", image_path)

    if not image_path:
        logger.error("No image path in state!")
        logger.info(_SECTION_SEP)
        return _error_response(
            "No image to analyze.",
            "I don't have an image to analyze. Please provide a URL.",
        )

    try:
        last_message = _get_last_user_message(state)
        base_prompt = "Analyze this landing page design."
        if last_message and not last_message.startswith(("http://", "https://", "Analyze this uploaded image:")):
            prompt = f"{base_prompt} Focus especially on the user's specific request/question: \"{last_message}\""
        else:
            prompt = base_prompt

        logger.info("Calling vision LLM for critique with prompt: %s", prompt)
        result = call_llm(
            prompt=prompt,
            system_prompt=UI_CRITIC_PROMPT,
            image_path=image_path,
            response_format=AnalysisResult,
        )

        report = _format_analysis_report(result)

        logger.info("UI Critic analysis complete.")
        logger.info(
            "Impression: %.80s …", result.overall_impression,
        )
        logger.info("Strengths: %d", len(result.strengths))
        logger.info("Critical issues: %d", len(result.critical_issues))
        logger.info(_SECTION_SEP)
        return {"analysis_report": report}
    except Exception as exc:
        logger.exception("Error in UI Critic")
        logger.info(_SECTION_SEP)
        return _error_response(
            str(exc), f"Error during analysis: {exc}",
        )


async def design_strategist_node(state: GraphState) -> Dict[str, Any]:
    """Create a design improvement plan from the UI Critic's report."""
    report = state.get("analysis_report")
    _log_section("🧠 RUNNING DESIGN_STRATEGIST NODE")

    if not report:
        logger.error("No analysis report in state!")
        logger.info(_SECTION_SEP)
        return {"current_error": "No analysis report available."}

    try:
        last_message = _get_last_user_message(state)
        base_prompt = "Create a design improvement plan."
        if last_message and not last_message.startswith(("http://", "https://", "Analyze this uploaded image:")):
            prompt = f"{base_prompt} Focus the strategy especially on addressing: \"{last_message}\""
        else:
            prompt = base_prompt

        logger.info("Generating design strategy plan with prompt: %s", prompt)
        result = call_llm(
            prompt=prompt,
            system_prompt=DESIGN_STRATEGIST_PROMPT.format(
                analysis_report=report,
            ),
            response_format=DesignPlan,
        )

        plan = _format_design_plan(result)

        logger.info("Design strategy complete.")
        logger.info("Overview: %.80s …", result.strategy_overview)
        logger.info("Layout items: %d", len(result.layout_improvements))
        logger.info("CTA items: %d", len(result.cta_optimization))
        logger.info(_SECTION_SEP)

        response_parts = [
            "## 🔍 UI/UX Analysis Report\n",
            report,
            "\n---\n## 🧠 Design Improvement Strategy\n",
            plan,
        ]

        has_image_key = any((
            os.getenv("GEMINI_API_KEY"),
            os.getenv("OPENAI_API_KEY"),
        ))
        if not has_image_key:
            response_parts.append(
                "\n---\n> ⚠️ **Image generation skipped** — "
                "no API key configured. Add a Gemini or OpenAI "
                "API key in the sidebar to enable visual redesign."
            )

        return {
            "design_plan": plan,
            "messages": [
                {"role": "assistant", "content": "\n".join(response_parts)},
            ],
        }
    except Exception as exc:
        logger.exception("Error in design strategist")
        logger.info(_SECTION_SEP)
        return _error_response(
            str(exc), f"Error during strategy planning: {exc}",
        )


async def visual_implementer_node(state: GraphState) -> Dict[str, Any]:
    """Generate an improved landing page image."""
    report = state.get("analysis_report")
    plan = state.get("design_plan")
    original_image = state.get("current_image_path")
    _log_section("🎨 RUNNING VISUAL_IMPLEMENTER NODE")
    logger.info("Reference image: %s", original_image)

    if not report or not plan:
        logger.error("Missing analysis_report or design_plan!")
        logger.info(_SECTION_SEP)
        return {"current_error": "Missing analysis or design plan."}

    try:
        last_message = _get_last_user_message(state)
        base_prompt = "Generate the detailed image prompt."
        if last_message and not last_message.startswith(("http://", "https://", "Analyze this uploaded image:")):
            prompt = f"{base_prompt} Ensure style adjustments incorporate: \"{last_message}\""
        else:
            prompt = base_prompt

        logger.info("Synthesising image-generation prompt with prompt: %s", prompt)
        image_gen_prompt = call_llm(
            prompt=prompt,
            system_prompt=VISUAL_IMPLEMENTER_PROMPT.format(
                analysis_report=report, design_plan=plan,
            ),
        )
        logger.info(
            "Prompt length: %d chars — generating image …",
            len(image_gen_prompt),
        )

        filepath = await generate_improved_landing_page(
            prompt=image_gen_prompt,
            reference_image_path=original_image,
        )
        logger.info("Improved design saved: %s", filepath)

        priorities_section = report.split("PRIORITIES:")[1].strip()
        response_msg = (
            "✅ **Analysis & Generation Complete!**\n\n"
            "I have analyzed your design, created an improvement "
            "strategy, and generated an updated version.\n\n"
            f"**Saved new design to:** `{filepath}`\n\n"
            f"**Key Priorities Addressed:**\n{priorities_section}\n"
        )

        logger.info(_SECTION_SEP)
        return {
            "messages": [{"role": "assistant", "content": response_msg}],
            "artifact_versions": {"latest_improved_design": 1},
            "artifact_filenames": {"latest_improved_design": filepath},
            "current_image_path": filepath,
        }
    except Exception as exc:
        logger.exception("Error in visual implementer")
        logger.info(_SECTION_SEP)
        return _error_response(
            str(exc), f"Error generating image: {exc}",
        )


async def design_editor_node(state: GraphState) -> Dict[str, Any]:
    """Edit an existing design based on user feedback."""
    last_message = _get_last_user_message(state)
    image_path = state.get("current_image_path")
    _log_section("✏️  RUNNING DESIGN_EDITOR NODE")
    logger.info("Edit request: %r", last_message)
    logger.info("Current image: %s", image_path)

    if not image_path:
        logger.warning("No current design to edit!")
        logger.info(_SECTION_SEP)
        return {
            "messages": [
                {
                    "role": "assistant",
                    "content": (
                        "There is no current design to edit. "
                        "Please analyze a page first."
                    ),
                },
            ],
        }

    try:
        prompt = f"Generate the edit prompt for the image generation model to satisfy the request: \"{last_message}\""
        logger.info("Generating edit prompt with prompt: %s", prompt)
        edit_prompt = call_llm(
            prompt=prompt,
            system_prompt=DESIGN_EDITOR_PROMPT.format(
                user_request=last_message,
            ),
        )
        logger.info("Edit prompt: %s", edit_prompt)

        filepath = await generate_improved_landing_page(
            prompt=edit_prompt, reference_image_path=image_path,
        )
        logger.info("Edited design saved: %s", filepath)

        current_version = (
            state.get("artifact_versions", {})
            .get("latest_improved_design", 1)
        )
        new_version = current_version + 1
        logger.info("Version %d → %d", current_version, new_version)

        response_msg = (
            "✅ **Design Edited Successfully!**\n\n"
            f"**Saved updated design to:** `{filepath}` "
            f"(Version {new_version})\n"
        )

        logger.info(_SECTION_SEP)
        return {
            "messages": [{"role": "assistant", "content": response_msg}],
            "artifact_versions": {"latest_improved_design": new_version},
            "artifact_filenames": {"latest_improved_design": filepath},
            "current_image_path": filepath,
        }
    except Exception as exc:
        logger.exception("Error during design edit")
        logger.info(_SECTION_SEP)
        return _error_response(
            str(exc), f"Error editing design: {exc}",
        )
