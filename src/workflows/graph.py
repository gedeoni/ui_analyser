from langgraph.graph import StateGraph, START, END
import os
import logging

logger = logging.getLogger(__name__)

from src.config.settings import settings
from src.state.schema import GraphState
from src.agents.nodes import (
    route_request,
    info_agent_node,
    capture_website_node,
    ui_critic_node,
    design_strategist_node,
    visual_implementer_node,
    design_editor_node
)

def build_graph():
    # Initialize the graph
    workflow = StateGraph(GraphState)

    # Add all nodes
    workflow.add_node("router", route_request)
    workflow.add_node("info_agent", info_agent_node)
    workflow.add_node("capture_website", capture_website_node)
    workflow.add_node("ui_critic", ui_critic_node)
    workflow.add_node("design_strategist", design_strategist_node)
    workflow.add_node("visual_implementer", visual_implementer_node)
    workflow.add_node("design_editor", design_editor_node)

    # Entry point
    workflow.add_edge(START, "router")

    # Conditional routing logic from the router
    def router_condition(state: GraphState) -> str:
        last_msg = state["messages"][-1]["content"]
        logger.info(f"Evaluating router condition from message: {repr(last_msg)}")
        if "capture_website" in last_msg:
            logger.info("Routing -> capture_website node")
            return "capture_website"
        elif "analysis_pipeline" in last_msg:
            logger.info("Routing -> ui_critic (analysis_pipeline) node")
            return "ui_critic"
        elif "design_editor" in last_msg:
            logger.info("Routing -> design_editor node")
            return "design_editor"
        logger.info("Routing -> info_agent node")
        return "info_agent"

    workflow.add_conditional_edges(
        "router",
        router_condition,
        {
            "capture_website": "capture_website",
            "ui_critic": "ui_critic",
            "design_editor": "design_editor",
            "info_agent": "info_agent"
        }
    )

    # Edge from capture website goes directly into analysis pipeline (ui_critic)
    workflow.add_edge("capture_website", "ui_critic")

    # Analysis pipeline is sequential
    workflow.add_edge("ui_critic", "design_strategist")

    # Conditional edge: only proceed to visual_implementer if image gen API keys exist
    def can_generate_image(state: GraphState) -> str:
        has_key = any([
            os.getenv("GEMINI_API_KEY"),
            os.getenv("OPENAI_API_KEY"),
            settings.GEMINI_API_KEY,
            settings.OPENAI_API_KEY,
        ])
        if has_key:
            logger.info("✅ Image generation API key found — proceeding to visual_implementer")
            return "visual_implementer"
        else:
            logger.warning("⚠️ No image generation API key — stopping at design_strategist")
            return "end"

    workflow.add_conditional_edges(
        "design_strategist",
        can_generate_image,
        {
            "visual_implementer": "visual_implementer",
            "end": END
        }
    )
    
    # End points
    workflow.add_edge("visual_implementer", END)
    workflow.add_edge("design_editor", END)
    workflow.add_edge("info_agent", END)

    return workflow

# Expose the workflow builder
workflow = build_graph()
