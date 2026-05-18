from langgraph.graph import StateGraph, START, END
import os

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
        # The router node sets the next node in its output (we can access it if we return it in the state, 
        # but LangGraph allows nodes to return dicts that update state. Wait, the router returns {"next_node": ...} 
        # but next_node isn't in our GraphState schema unless we add it, or we can use the last system message.
        # Let's adjust this: the router can just return the route string directly if we use it as a conditional edge,
        # but since route_request updates state, we need to inspect the state.
        
        # Let's parse the last system message which contains "Routed to: X"
        last_msg = state["messages"][-1]["content"]
        if "capture_website" in last_msg:
            return "capture_website"
        elif "analysis_pipeline" in last_msg:
            return "ui_critic"
        elif "design_editor" in last_msg:
            return "design_editor"
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
    workflow.add_edge("design_strategist", "visual_implementer")
    
    # End points
    workflow.add_edge("visual_implementer", END)
    workflow.add_edge("design_editor", END)
    workflow.add_edge("info_agent", END)

    return workflow

# Expose the workflow builder
workflow = build_graph()
