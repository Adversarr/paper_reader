# tag_manager.py
import os
from typing import Iterable, List, Dict, Optional
from paper_reader.models import TagInfo, ArticleSummary, Content
from paper_reader.prompts import (
    TAG_PRUNE_SYSTEM,
    TAG_SURVEY_SYSTEM,
    create_message_entry,
)
from paper_reader.utils import (
    ensure_dir_exists,
    slugify,
    save_text_and_embedding,
    load_text_and_embedding,
)
from paper_reader.config import (
    MODEL_FAST,
    MODEL_INSTRUCT,
    TAGS_DIR,
    TAG_DESCRIPTION_MD_FILE,
    TAG_SURVEY_MD_FILE,
    MAX_TOKENS_TAG_DESCRIPTION,
    DOCS_DIR,
    SUMMARIZED_MD_FILE,
    MAX_TOKENS_TAG_ARTICLE,
    LOGGER,
)
from paper_reader.openai_utils import (
    generate_completion,
    get_embedding,
)
from paper_reader.vector_store import get_relevant_context_for_prompt

# Global store for tags (in-memory, could be persisted to a JSON file)
# Maps tag_slug to TagInfo
# This should ideally be loaded from disk at startup and saved on modification
# For simplicity, we'll manage it in memory per run, relying on file system for persistence of MD/NPZ.
# A more robust approach would be a `load_all_tags_from_disk` function.
_GLOBAL_TAG_STORE: Dict[str, TagInfo] = {}


def _str_to_list(s: str) -> List[str]:
    # 1. remove all the none [0-9a-zA-Z] or `-` charactors in string
    pruned_str = "".join([c if c.isalnum() or c in "-," else "" for c in s])
    return [i.strip() for i in pruned_str.split(",") if i.strip()]


def get_or_create_tag_info(tag_name: str) -> TagInfo:
    """
    Retrieves existing TagInfo from disk or creates a new one.
    This function now primarily loads from disk.
    """
    tag_slug = slugify(tag_name)
    tag_dir = os.path.join(TAGS_DIR, tag_slug)
    ensure_dir_exists(tag_dir)

    if tag_slug in _GLOBAL_TAG_STORE:
        return _GLOBAL_TAG_STORE[tag_slug]

    description_content: Optional[Content] = load_text_and_embedding(tag_dir, TAG_DESCRIPTION_MD_FILE)
    survey_content: Optional[Content] = load_text_and_embedding(tag_dir, TAG_SURVEY_MD_FILE)

    # For related_paper_slugs, this would ideally be stored in a meta file for the tag.
    # For this minimal version, we'll rebuild it if needed or leave it empty on initial load.
    # It gets populated when update_tag_with_article is called.
    # To truly persist, we'd need a tag_meta.json or similar.

    # If tag_slug is in _GLOBAL_TAG_STORE, use its related_paper_slugs, otherwise initialize.
    # This part needs careful state management if mixing in-memory and disk.
    # For now, let's assume related_paper_slugs are primarily managed by `update_tag_with_article`.
    # A robust load would scan all articles and rebuild this mapping.

    related_papers = []
    if tag_slug in _GLOBAL_TAG_STORE:
        related_papers = _GLOBAL_TAG_STORE[tag_slug]["related_paper_slugs"]

    tag_info = TagInfo(
        name=tag_name,
        tag_slug=tag_slug,
        description=description_content,
        survey=survey_content,
        related_paper_slugs=related_papers,
    )
    _GLOBAL_TAG_STORE[tag_slug] = tag_info  # Update global store
    return tag_info


async def generate_tag_description(tag_name: str, force_regenerate: bool = False) -> Optional[Content]:
    """Generates a description for a tag using an LLM."""
    tag_slug = slugify(tag_name)
    tag_dir = os.path.join(TAGS_DIR, tag_slug)
    ensure_dir_exists(tag_dir)
    
    # Check if description already exists and we're not forcing regeneration
    if not force_regenerate:
        existing_description = load_text_and_embedding(tag_dir, TAG_DESCRIPTION_MD_FILE)
        if existing_description:
            return existing_description
    
    # Use RAG to get relevant context
    rag_query = f"What is {tag_name} in academic context?"
    rag_context = get_relevant_context_for_prompt(rag_query, "tags")
    
    # Build prompt
    prompt = [
        create_message_entry(
            "user",
            f"Generate a comprehensive description of the research area, concept, or technology known as '{tag_name}'.\n\n"
            "This should be a well-structured, informative explanation suitable for academic context."
        )
    ]
    
    if rag_context and rag_context != "No existing content to retrieve from.":
        prompt.append(
            create_message_entry(
                "user",
                f"Here is some relevant context that might help:\n\n{rag_context}"
            )
        )
    
    # Generate the description
    description_text = await generate_completion(
        prompt=prompt,
        system_prompt="You are an expert academic researcher with deep knowledge across many fields. "
                     "Provide detailed, accurate descriptions of academic concepts.",
        model=MODEL_INSTRUCT,
        max_tokens=MAX_TOKENS_TAG_DESCRIPTION,
        temperature=0.3,  # Lower temperature for more factual output
    )
    
    if not description_text:
        LOGGER.error(f"Failed to generate description for tag: {tag_name}")
        return None
    
    # Generate embedding for the description
    embedding = get_embedding(description_text)
    
    # Save the description and its embedding
    description_content = Content(content=description_text, vector=embedding)
    save_text_and_embedding(tag_dir, TAG_DESCRIPTION_MD_FILE, description_text, embedding)
    
    return description_content


async def aprocess_tag(tag_info: TagInfo):
    """Generates the description and survey for a tag."""
    tag_name = tag_info["name"]
    tag_slug = tag_info["tag_slug"]
    tag_dir = os.path.join(TAGS_DIR, tag_slug)
    
    # Generate description if needed
    if tag_info["description"] is None:
        LOGGER.info(f"Generating description for tag: {tag_name}")
        description_content = await generate_tag_description(tag_name)
        if description_content:
            tag_info["description"] = description_content
            save_text_and_embedding(
                tag_dir, TAG_DESCRIPTION_MD_FILE, description_content["content"], description_content["vector"]
            )
    
    # Generate survey if needed
    if tag_info["survey"] is None and tag_info["related_paper_slugs"]:
        LOGGER.info(f"Generating survey for tag: {tag_name}")
        survey_content = await agenerate_tag_survey(tag_name)
        if survey_content:
            tag_info["survey"] = survey_content
            save_text_and_embedding(
                tag_dir, TAG_SURVEY_MD_FILE, survey_content["content"], survey_content["vector"]
            )


async def agenerate_tag_survey(tag_name: str, force_regenerate: bool = False) -> Optional[Content]:
    """Generates a survey of papers for a specific tag."""
    tag_slug = slugify(tag_name)
    tag_info = get_or_create_tag_info(tag_name)
    tag_dir = os.path.join(TAGS_DIR, tag_slug)
    ensure_dir_exists(tag_dir)
    
    # Check if survey already exists and we're not forcing regeneration
    if not force_regenerate and tag_info["survey"] is not None:
        return tag_info["survey"]
    
    # If there are no related papers, we can't generate a survey
    if not tag_info["related_paper_slugs"]:
        LOGGER.warning(f"No related papers found for tag: {tag_name}. Cannot generate survey.")
        return None
    
    # Gather summaries of related papers
    paper_summaries = []
    for paper_slug in tag_info["related_paper_slugs"]:
        paper_dir = os.path.join(DOCS_DIR, paper_slug)
        summary = load_text_and_embedding(paper_dir, SUMMARIZED_MD_FILE)
        if summary:
            paper_summaries.append(summary["content"])
    
    if not paper_summaries:
        LOGGER.warning(f"No paper summaries found for tag: {tag_name}. Cannot generate survey.")
        return None
    
    # Build prompt for the survey
    paper_context_text = "\n\n---\n".join(paper_summaries)
    prompt = [
        create_message_entry(
            "user",
            f"Generate a comprehensive survey of research papers related to '{tag_name}'.\n\n"
            f"Here are the summaries of relevant papers:\n\n{paper_context_text}\n\n"
            "Provide a well-structured survey that synthesizes the key findings, methodologies, "
            "and contributions of these papers, highlighting relationships between them."
        )
    ]
    
    # Generate the survey
    survey_text = await generate_completion(
        prompt=prompt,
        system_prompt=TAG_SURVEY_SYSTEM,
        model=MODEL_INSTRUCT,
        max_tokens=MAX_TOKENS_TAG_ARTICLE,
        temperature=0.4,
    )
    
    if not survey_text:
        LOGGER.error(f"Failed to generate survey for tag: {tag_name}")
        return None
    
    # Generate embedding for the survey
    embedding = get_embedding(survey_text)
    
    # Save the survey and its embedding
    survey_content = Content(content=survey_text, vector=embedding)
    save_text_and_embedding(tag_dir, TAG_SURVEY_MD_FILE, survey_text, embedding)
    
    # Update tag info in memory
    tag_info["survey"] = survey_content
    
    return survey_content


async def prune_tags(tags: Iterable[str]) -> List[str]:
    """Filters and standardizes a list of tags using LLM."""
    # If there are no tags, return an empty list
    if not tags:
        return []
    
    # Convert to list and deduplicate
    tag_list = list(set(tags))
    
    # If there's only one or two tags, no need to call the LLM
    if len(tag_list) <= 2:
        return tag_list
    
    # Prepare tag list as a comma-separated string
    tags_str = ", ".join(tag_list)
    
    # Build prompt for tag pruning
    prompt = [
        create_message_entry(
            "user",
            f"I have extracted the following tags from an academic paper: {tags_str}\n\n"
            "Please standardize, consolidate, and filter these tags to ensure they are:\n"
            "1. Relevant academic concepts or research areas\n"
            "2. Not too general or too specific\n"
            "3. Consistent in format and terminology\n\n"
            "Return only the final list of tags as a comma-separated list, with no explanation."
        )
    ]
    
    # Generate the pruned tag list
    pruned_tags_str = await generate_completion(
        prompt=prompt,
        system_prompt=TAG_PRUNE_SYSTEM,
        model=MODEL_FAST,  # Use a faster model for this task
        max_tokens=200,  # Reasonable limit for tag lists
        temperature=0.2,  # Lower temperature for more consistent output
    )
    
    if not pruned_tags_str:
        LOGGER.warning("Failed to prune tags, returning original list")
        return tag_list
    
    # Parse the pruned tags back into a list
    pruned_tags = [tag.strip() for tag in pruned_tags_str.split(",") if tag.strip()]
    
    # If parsing failed, return the original list
    if not pruned_tags:
        return tag_list
    
    return pruned_tags


async def update_tag_with_article(tag_name: str, article: ArticleSummary):
    """Updates a tag with information from an article."""
    tag_info = get_or_create_tag_info(tag_name)
    paper_slug = article["paper_slug"]
    
    # If the paper is already associated with the tag, nothing to do
    if paper_slug in tag_info["related_paper_slugs"]:
        return tag_info
    
    # Add paper slug to the tag's related papers
    tag_info["related_paper_slugs"].append(paper_slug)
    
    # If this is the first paper for this tag, generate the description
    if len(tag_info["related_paper_slugs"]) == 1 and tag_info["description"] is None:
        description_content = await generate_tag_description(tag_name)
        if description_content:
            tag_info["description"] = description_content
            tag_dir = os.path.join(TAGS_DIR, tag_info["tag_slug"])
            save_text_and_embedding(
                tag_dir, TAG_DESCRIPTION_MD_FILE, description_content["content"], description_content["vector"]
            )
    
    # If we have enough papers and no survey yet, generate the survey
    # This could be tuned; here we generate after 3 papers
    if len(tag_info["related_paper_slugs"]) >= 3 and tag_info["survey"] is None:
        survey_content = await agenerate_tag_survey(tag_name)
        if survey_content:
            tag_info["survey"] = survey_content
    
    return tag_info


async def update_tags(article: ArticleSummary, pruned_tags: List[str]):
    """Updates all tags for an article."""
    LOGGER.info(f"Updating tags for article: {article['title']}")
    
    for tag_name in pruned_tags:
        await update_tag_with_article(tag_name, article)
    
    # Update article with pruned tags
    article["tags"] = pruned_tags


async def process_all_tags_iteratively(all_articles: List[ArticleSummary]):
    """Processes all tags based on a list of articles."""
    LOGGER.info("Processing all tags...")
    
    # Clear global tag store to rebuild from disk
    _GLOBAL_TAG_STORE.clear()
    
    # Process each article's tags
    for article in all_articles:
        if not article:  # Skip None articles
            continue
            
        tags = article.get("tags", [])
        if tags:
            await update_tags(article, tags)
    
    # Process all tags that have articles
    for tag_slug, tag_info in _GLOBAL_TAG_STORE.items():
        if tag_info["related_paper_slugs"]:
            await aprocess_tag(tag_info)
    
    LOGGER.info(f"Finished processing {len(_GLOBAL_TAG_STORE)} tags")
