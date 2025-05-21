# vector_store.py
import numpy as np
import os
from typing import List, Tuple, Optional
from paper_reader.models import Content, ArticleSummary, TagInfo
from paper_reader.utils import load_text_and_embedding
from paper_reader.config import DOCS_DIR, TAGS_DIR, SUMMARIZED_MD_FILE, TAG_DESCRIPTION_MD_FILE

# In a real vector database, this would be more sophisticated.
# For this minimal implementation, we'll load embeddings on demand for querying.
# We assume embeddings are already stored alongside their .md files.


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Computes cosine similarity between two vectors."""
    if vec1 is None or vec2 is None:
        return 0.0
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0.0
    return dot_product / (norm_vec1 * norm_vec2)


def find_similar_content(query_embedding: np.ndarray, content_list: List[Content], top_k: int = 5) -> List[Content]:
    """Finds top_k similar content items from a list based on cosine similarity."""
    if query_embedding is None or not content_list:
        return []

    similarities = []
    for item in content_list:
        if item["vector"] is not None:
            sim = cosine_similarity(query_embedding, item["vector"])
            similarities.append((item, sim))

    similarities.sort(key=lambda x: x[1], reverse=True)
    return [item for item, sim in similarities[:top_k]]


def load_all_article_summaries_for_rag() -> List[Content]:
    """Loads all available article summaries (summarized.md) for RAG."""
    all_summaries: List[Content] = []
    if not os.path.exists(DOCS_DIR):
        return []
    for paper_slug in os.listdir(DOCS_DIR):
        paper_dir = os.path.join(DOCS_DIR, paper_slug)
        if os.path.isdir(paper_dir):
            summary_content = load_text_and_embedding(paper_dir, SUMMARIZED_MD_FILE)
            if summary_content and summary_content["content"]:  # Ensure content is not empty
                # Add title information for better context in RAG prompts
                title_from_slug = paper_slug.replace("-", " ").title()
                enriched_content = f"Article Title: {title_from_slug}\nSummary:\n{summary_content['content']}"
                all_summaries.append(Content(content=enriched_content, vector=summary_content["vector"]))
    return all_summaries


def load_all_tag_descriptions_for_rag() -> List[Content]:
    """Loads all available tag descriptions for RAG."""
    all_descriptions: List[Content] = []
    if not os.path.exists(TAGS_DIR):
        return []
    for tag_slug in os.listdir(TAGS_DIR):
        tag_dir = os.path.join(TAGS_DIR, tag_slug)
        if os.path.isdir(tag_dir):
            desc_content = load_text_and_embedding(tag_dir, TAG_DESCRIPTION_MD_FILE)
            if desc_content and desc_content["content"]:
                tag_name_from_slug = tag_slug.replace("-", " ").title()
                enriched_content = f"Tag: {tag_name_from_slug}\nDescription:\n{desc_content['content']}"
                all_descriptions.append(Content(content=enriched_content, vector=desc_content["vector"]))
    return all_descriptions


# Example of how RAG might be used (conceptual, actual use is in generation prompts)
def get_relevant_context_for_prompt(
    query_text: str,
    source_type: str = "articles",  # "articles" or "tags"
    top_k: int = 3,
) -> str:
    """
    Retrieves relevant context from stored summaries or descriptions.
    This is a helper to build parts of a RAG prompt.
    """
    from paper_reader.openai_utils import get_embedding  # Avoid circular import at top level

    query_embedding = get_embedding(query_text)
    if query_embedding is None:
        return "No relevant context found (embedding generation failed)."

    candidate_content: List[Content] = []
    if source_type == "articles":
        candidate_content = load_all_article_summaries_for_rag()
    elif source_type == "tags":
        candidate_content = load_all_tag_descriptions_for_rag()
    else:
        return "Invalid source type for RAG."

    if not candidate_content:
        return "No existing content to retrieve from."

    similar_items = find_similar_content(query_embedding, candidate_content, top_k)

    if not similar_items:
        return "No similar content found."

    context_str = f"\n\n---\nRelevant Information ({source_type}):\n---\n"
    for item in similar_items:
        context_str += item["content"] + "\n---\n"

    return context_str
