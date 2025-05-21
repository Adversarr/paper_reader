# models.py
from typing import TypedDict, List, Optional
import numpy as np


class Content(TypedDict):
    """Represents a piece of text content and its corresponding vector embedding."""

    content: str
    vector: Optional[
        np.ndarray
    ]  # Vector can be None if not yet computed or not applicable


class ArticleSummary(TypedDict):
    """Holds all processed information for a single article."""

    title: str  # Original title, will be slugified for directory name
    paper_slug: str  # Filesystem-safe name derived from title
    content_path: str  # Path to extracted.md
    content: Content  # Original content from extracted.md
    summary: Optional[Content]  # Article-level summary
    tldr: Optional[Content]  # TLDR summary
    tags: List[str]  # List of extracted tag names (strings)


class TagInfo(TypedDict):
    """Holds information about a specific tag."""

    name: str  # Original tag name
    tag_slug: str  # Filesystem-safe name derived from tag name
    description: Optional[Content]  # Wikipedia-like description of the tag
    survey: Optional[Content]  # Survey of related papers for this tag
    related_paper_slugs: List[str]  # List of paper slugs associated with this tag
