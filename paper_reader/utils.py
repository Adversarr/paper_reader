# utils.py
import os
import re
from typing_extensions import deprecated
import numpy as np
from typing import Optional, List
from paper_reader.models import Content


def ensure_dir_exists(path: str):
    """Ensures that a directory exists, creating it if necessary."""
    os.makedirs(path, exist_ok=True)


def slugify(text: str) -> str:
    """Converts text to a filesystem-friendly slug."""
    text = text.lower()
    text = re.sub(r"\s+", "-", text)  # Replace spaces with hyphens
    text = re.sub(
        r"[^\w\-]", "", text
    )  # Remove non-alphanumeric characters except hyphens
    text = text.strip("-")
    return text if text else "untitled"

def human_readable_slugify(text: str) -> str:
    return text.replace("-", " ").replace("_", " ").title()

def save_text_and_embedding(
    base_path: str,
    filename_md: str,
    text_content: str,
    embedding_vector: Optional[np.ndarray],
):
    """Saves the text content to a .md file and its embedding to a .npz file."""
    ensure_dir_exists(
        os.path.dirname(base_path)
    )  # Ensure directory for base_path itself
    md_path = os.path.join(base_path, filename_md)
    npz_path = os.path.join(base_path, filename_md.replace(".md", ".npz"))

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(text_content)

    if embedding_vector is not None:
        np.savez_compressed(npz_path, vector=embedding_vector)


def load_text_and_embedding(base_path: str, filename_md: str) -> Optional[Content]:
    """Loads text content and its embedding. Returns None if files don't exist."""
    md_path = os.path.join(base_path, filename_md)
    npz_path = os.path.join(base_path, filename_md.replace(".md", ".npz"))

    if not os.path.exists(md_path):
        return None

    with open(md_path, "r", encoding="utf-8") as f:
        text_content = f.read()

    embedding_vector = None
    if os.path.exists(npz_path):
        try:
            data = np.load(npz_path)
            embedding_vector = data["vector"]
        except Exception as e:
            print(f"Warning: Could not load embedding from {npz_path}: {e}")
            embedding_vector = None  # Or handle more gracefully

    return Content(content=text_content, vector=embedding_vector)

@deprecated("No longer needed")
def split_into_sections(text: str, separator: str | None) -> List[str]:
    """Splits text into sections based on a separator."""
    if separator is None:
        return split_into_sections_use_markdown_heading2(text=text)
    if separator == '':
        return split_into_sections_use_markdown_heading2(text=text)
    if separator in text:
        return [section.strip() for section in text.split(separator) if section.strip()]
    return [text.strip()]  # Return the whole text as one section if separator not found

@deprecated("No longer needed")
def split_into_sections_use_markdown_heading2(text: str) -> List[str]:
    """Splits text into sections using Markdown heading level 2 (`## Heading`)."""
    import re

    # Split at lines that begin with "## "
    sections = re.split(r"(?m)^## ", text)
    all_sections = [section.strip() for section in sections if section.strip()]
    # Merge the front two if valid
    if len(all_sections) > 1 and not all_sections[0].startswith("##"):
        all_sections[0] = all_sections[0] + "\n" + all_sections.pop(1)

    # Add the heading 2 text to the section content
    for i in range(1, len(all_sections)):
        if not all_sections[i].startswith("## "):
            all_sections[i] = "## " + all_sections[i]

    return all_sections
