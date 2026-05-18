"""LangGraph workflow definition for the UI Analyser agent team.

Builds a ``StateGraph`` with routing, analysis pipeline, image
generation, and design-editing nodes.
"""

import logging
import os

from langgraph.graph import END, START, StateGraph

from src.agents.nodes import (
    capture_website_node,
    design_editor_node,
    design_strategist_node,
    info_agent_node,
    route_request,
    ui_critic_node,
    visual_implementer_node,
)
from src.config.settings import settings
from src.state.schema import GraphState

logger = logging.getLogger(__name__)


def build_graph() -> StateGraph:
    """Construct and return the (uncompiled) agent workflow graph."""
    wf = StateGraph(GraphState)

    # Register nodes
    wf.add_node("router", route_request)
    wf.add_node("info_agent", info_agent_node)
    wf.add_node("capture_website", capture_website_node)
    wf.add_node("ui_critic", ui_critic_node)
    wf.add_node("design_strategist", design_strategist_node)
    wf.add_node("visual_implementer", visual_implementer_node)
    wf.add_node("design_editor", design_editor_node)

    # Entry point
    wf.add_edge(START, "router")

    # Router → conditional fan-out
    wf.add_conditional_edges(
        "router",
        _router_condition,
        {
            "capture_website": "capture_website",
            "ui_critic": "ui_critic",
            "design_editor": "design_editor",
            "info_agent": "info_agent",
        },
    )

    # Analysis pipeline: capture → critic → strategist → (maybe) visual
    wf.add_edge("capture_website", "ui_critic")
    wf.add_edge("ui_critic", "design_strategist")

    wf.add_conditional_edges(
        "design_strategist",
        _can_generate_image,
        {
            "visual_implementer": "visual_implementer",
            "end": END,
        },
    )

    # Terminal edges
    wf.add_edge("visual_implementer", END)
    wf.add_edge("design_editor", END)
    wf.add_edge("info_agent", END)

    return wf


# ── Condition helpers ───────────────────────────────────────────────

def _router_condition(state: GraphState) -> str:
    """Map the router's system message to a downstream node name."""
    last_msg = state["messages"][-1]["content"]
    logger.info("Router condition — message: %r", last_msg)

    route_map = {
        "capture_website": "capture_website",
        "analysis_pipeline": "ui_critic",
        "design_editor": "design_editor",
    }
    for keyword, target in route_map.items():
        if keyword in last_msg:
            logger.info("Routing → %s", target)
            return target

    logger.info("Routing → info_agent (default)")
    return "info_agent"


def _can_generate_image(state: GraphState) -> str:  # noqa: ARG001  # pylint: disable=unused-argument
    """Check whether an image-generation API key is available."""
    has_key = any((
        os.getenv("GEMINI_API_KEY"),
        os.getenv("OPENAI_API_KEY"),
        settings.GEMINI_API_KEY,
        settings.OPENAI_API_KEY,
    ))
    if has_key:
        logger.info("Image generation key found → visual_implementer")
        return "visual_implementer"

    logger.warning("No image generation key → stopping at strategist")
    return "end"


# Expose the workflow builder as a module-level object.
workflow = build_graph()
