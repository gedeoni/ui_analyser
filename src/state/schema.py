"""State schema definitions for the UI Analyser agent graph.

Defines the shared graph state (GraphState), reducer functions for
list/dict fields, and Pydantic models for structured LLM outputs.
"""

from typing import Annotated, Any, Dict, List, Optional, TypedDict

from pydantic import BaseModel, Field


# ── Reducer functions (append-based merging for LangGraph) ──────────


def add_messages(left: list, right: list) -> list:
    """Append new messages to the existing message list."""
    return left + right


def update_dict(left: dict, right: dict) -> dict:
    """Shallow-merge *right* into a copy of *left*."""
    updated = left.copy()
    updated.update(right)
    return updated


class GraphState(TypedDict):
    """The state of the UI Analyser Agent Graph."""

    # Conversation history
    messages: Annotated[List[Dict[str, Any]], add_messages]

    # Context specific to the current task
    target_url: Optional[str]
    current_image_path: Optional[str]

    # Extracted data from the analysis pipeline
    analysis_report: Optional[str]
    design_plan: Optional[str]

    # Artifacts (tracking generated images and their versions)
    artifact_versions: Annotated[Dict[str, int], update_dict]
    artifact_filenames: Annotated[Dict[str, str], update_dict]

    # For tracking error or status states
    current_error: Optional[str]


# ── Pydantic models for structured LLM outputs ─────────────────────


class AnalysisResult(BaseModel):
    """Structured output for the UI Critic."""

    overall_impression: str = Field(
        description="Rating and 2-3 sentence summary",
    )
    strengths: List[str] = Field(
        description="List of 3-5 things that work well",
    )
    critical_issues: List[str] = Field(
        description="List of high priority issues",
    )
    additional_improvements: List[str] = Field(
        description="Medium/Low priority suggestions",
    )
    top_3_priorities: List[str] = Field(
        description="Top 3 most impactful changes",
    )
    scores: Dict[str, int] = Field(
        description=(
            "Scores out of 10 for Layout, Typography, "
            "Color, CTA, Whitespace"
        ),
    )


class TypographySpec(BaseModel):
    """Explicit typography specification."""

    font_family: str = Field(
        description="Selected Google or web-safe font pairing, e.g., 'Inter'",
    )
    base_font_size: str = Field(
        description="Body text size with unit, e.g., '16px'",
    )
    line_height_ratio: float = Field(
        description=(
            "Line height decimal multiplier. MUST be between 1.4 "
            "and 1.6 for body copy readability (e.g., 1.5)"
        ),
    )
    letter_spacing: str = Field(
        description=(
            "Letter spacing with unit. MUST be positive or 'normal' "
            "for body text (e.g., '0.01em', 'normal')"
        ),
    )


class ColorPaletteSpec(BaseModel):
    """Strict, accessible color palette specifications."""

    primary_cta_bg: str = Field(
        description="Hex code for primary CTA button background",
    )
    primary_cta_text: str = Field(
        description="Hex code for CTA button text (must pass 4.5:1 contrast against primary_cta_bg)",
    )
    background: str = Field(
        description="Hex code for the primary page background",
    )
    text_color: str = Field(
        description="Hex code for body text (must pass 4.5:1 contrast against background)",
    )
    accent: str = Field(
        description="Hex code for visual highlights or subtle secondary accents",
    )


class DesignPlan(BaseModel):
    """Structured output for the Design Strategist."""

    strategy_overview: str
    layout_improvements: List[str]
    color_palette: ColorPaletteSpec = Field(
        description="Strict, accessible color palette specifications",
    )
    typography: TypographySpec = Field(
        description="Strict typographical specs",
    )
    cta_optimization: List[str]
    accessibility_enhancements: List[str]
    mobile_considerations: List[str]
    content_recommendations: List[str]
