# tag_manager.py
import os
from typing import List, Dict, Optional
from paper_reader.models import TagInfo, ArticleSummary, Content
from paper_reader.utils import (
    ensure_dir_exists,
    slugify,
    save_text_and_embedding,
    load_text_and_embedding
)
from paper_reader.config import (
    TAGS_DIR,
    TAG_DESCRIPTION_MD_FILE,
    TAG_SURVEY_MD_FILE,
    PROMPT_TAG_DESCRIPTION,
    PROMPT_TAG_SURVEY,
    MAX_TOKENS_TAG_DESCRIPTION,
    MAX_TOKENS_TAG_SURVEY,
    DOCS_DIR, SUMMARIZED_MD_FILE
)
from paper_reader.openai_utils import get_embedding, generate_completion
from paper_reader.vector_store import get_relevant_context_for_prompt # For RAG

# Global store for tags (in-memory, could be persisted to a JSON file)
# Maps tag_slug to TagInfo
# This should ideally be loaded from disk at startup and saved on modification
# For simplicity, we'll manage it in memory per run, relying on file system for persistence of MD/NPZ.
# A more robust approach would be a `load_all_tags_from_disk` function.
_GLOBAL_TAG_STORE: Dict[str, TagInfo] = {}


def get_or_create_tag_info(tag_name: str) -> TagInfo:
    """
    Retrieves existing TagInfo from disk or creates a new one.
    This function now primarily loads from disk.
    """
    tag_slug = slugify(tag_name)
    tag_dir = os.path.join(TAGS_DIR, tag_slug)
    ensure_dir_exists(tag_dir)

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
    if tag_slug in _GLOBAL_TAG_STORE: # If we have some in-memory state
        related_papers = _GLOBAL_TAG_STORE[tag_slug].get('related_paper_slugs', [])


    tag_info = TagInfo(
        name=tag_name,
        tag_slug=tag_slug,
        description=description_content,
        survey=survey_content,
        related_paper_slugs=related_papers # This will be updated
    )
    _GLOBAL_TAG_STORE[tag_slug] = tag_info # Update global store
    return tag_info


def update_tag_with_article(tag_name: str, article: ArticleSummary):
    """
    Updates a tag's information based on a newly processed or relevant article.
    This function will manage adding the article to the tag's list and can trigger
    regeneration of description or survey.
    """
    tag_slug = slugify(tag_name)
    tag_info = get_or_create_tag_info(tag_name) # Loads or initializes

    if article['paper_slug'] not in tag_info['related_paper_slugs']:
        tag_info['related_paper_slugs'].append(article['paper_slug'])
        # Persist this change if _GLOBAL_TAG_STORE is saved to disk, or manage via file markers.
        # For now, it's in-memory for the current run.
        print(f"Article '{article['title']}' added to tag '{tag_name}'.")
        # Trigger regeneration of description and survey as a new paper is relevant
        generate_tag_description(tag_name, force_regenerate=True)
        generate_tag_survey(tag_name, force_regenerate=True)
    
    _GLOBAL_TAG_STORE[tag_slug] = tag_info # Ensure global store is updated


def generate_tag_description(tag_name: str, force_regenerate: bool = False):
    """
    Generates (or re-generates) a Wikipedia-like description for a tag.
    Uses RAG by fetching summaries of related articles.
    """
    tag_info = get_or_create_tag_info(tag_name)
    tag_slug = tag_info['tag_slug']
    tag_dir = os.path.join(TAGS_DIR, tag_slug)

    if not force_regenerate and tag_info['description'] and tag_info['description']['content']:
        print(f"Description for tag '{tag_name}' already exists. Skipping generation.")
        return

    print(f"Generating description for tag: {tag_name}")

    related_article_summaries_text = ""
    # Collect summaries of papers associated with this tag
    # This requires access to ArticleSummary objects or their stored summaries.
    # For simplicity, load them directly.
    # In a larger system, you might query a database or a central article store.
    
    # Build context from related papers
    context_summaries = []
    for paper_slug in tag_info.get('related_paper_slugs', []):
        paper_dir = os.path.join(DOCS_DIR, paper_slug)
        summary_content_obj = load_text_and_embedding(paper_dir, SUMMARIZED_MD_FILE)
        if summary_content_obj and summary_content_obj['content']:
            # Add title for clarity
            title = paper_slug.replace('-', ' ').title()
            context_summaries.append(f"Article: {title}\nSummary: {summary_content_obj['content']}")
    
    if context_summaries:
        related_article_summaries_text = "\n\n".join(context_summaries[:5]) # Limit context size
    else:
        related_article_summaries_text = "No specific related articles found yet for context."

    previous_description_text = tag_info['description']['content'] if tag_info['description'] else "N/A"

    prompt = PROMPT_TAG_DESCRIPTION.format(
        tag_name=tag_name,
        related_article_summaries=related_article_summaries_text,
        previous_description=previous_description_text
    )

    description_text = generate_completion(prompt, max_tokens=MAX_TOKENS_TAG_DESCRIPTION)
    if description_text:
        embedding_vector = get_embedding(description_text)
        save_text_and_embedding(tag_dir, TAG_DESCRIPTION_MD_FILE, description_text, embedding_vector)
        tag_info['description'] = Content(content=description_text, vector=embedding_vector)
        _GLOBAL_TAG_STORE[tag_slug] = tag_info # Update store
        print(f"  Generated and saved description for tag '{tag_name}'.")
    else:
        print(f"  Failed to generate description for tag '{tag_name}'.")


def generate_tag_survey(tag_name: str, force_regenerate: bool = False):
    """
    Generates (or re-generates) a survey of related papers for a tag.
    """
    tag_info = get_or_create_tag_info(tag_name)
    tag_slug = tag_info['tag_slug']
    tag_dir = os.path.join(TAGS_DIR, tag_slug)

    if not force_regenerate and tag_info['survey'] and tag_info['survey']['content']:
        print(f"Survey for tag '{tag_name}' already exists. Skipping generation.")
        return

    print(f"Generating survey for tag: {tag_name}")

    # Need tag description for context
    tag_description_text = "N/A"
    if tag_info['description'] and tag_info['description']['content']:
        tag_description_text = tag_info['description']['content']
    elif os.path.exists(os.path.join(tag_dir, TAG_DESCRIPTION_MD_FILE)): # try to load if not in memory
        desc_obj = load_text_and_embedding(tag_dir, TAG_DESCRIPTION_MD_FILE)
        if desc_obj: tag_description_text = desc_obj['content']


    related_articles_info_text = ""
    article_infos = []
    for paper_slug in tag_info.get('related_paper_slugs', []):
        paper_dir = os.path.join(DOCS_DIR, paper_slug)
        summary_content_obj = load_text_and_embedding(paper_dir, SUMMARIZED_MD_FILE)
        if summary_content_obj and summary_content_obj['content']:
            title = paper_slug.replace('-', ' ').title() # Get title from slug
            article_infos.append(f"Paper Title: {title}\nSummary: {summary_content_obj['content']}")

    if article_infos:
        related_articles_info_text = "\n\n---\n\n".join(article_infos)
    else:
        related_articles_info_text = "No related articles found to include in the survey."

    previous_survey_text = tag_info['survey']['content'] if tag_info['survey'] else "N/A"

    prompt = PROMPT_TAG_SURVEY.format(
        tag_name=tag_name,
        tag_description=tag_description_text,
        related_articles_info=related_articles_info_text,
        previous_survey=previous_survey_text
    )

    survey_text = generate_completion(prompt, max_tokens=MAX_TOKENS_TAG_SURVEY)
    if survey_text:
        embedding_vector = get_embedding(survey_text)
        save_text_and_embedding(tag_dir, TAG_SURVEY_MD_FILE, survey_text, embedding_vector)
        tag_info['survey'] = Content(content=survey_text, vector=embedding_vector)
        _GLOBAL_TAG_STORE[tag_slug] = tag_info # Update store
        print(f"  Generated and saved survey for tag '{tag_name}'.")
    else:
        print(f"  Failed to generate survey for tag '{tag_name}'.")

def process_all_tags_iteratively(all_articles: List[ArticleSummary]):
    """
    Iteratively updates all tags based on the full list of articles.
    This function is called after all articles are processed.
    It ensures that tag descriptions and surveys are (re)generated
    with the full context of all related papers.
    """
    print("\n--- Updating All Tags Iteratively ---")
    
    # First, ensure all tags from all articles are known and articles are associated
    for article in all_articles:
        if article and article.get('tags'):
            for tag_name_slug in article['tags']: # Assuming tags are already slugified
                # Convert slug back to a more readable name if possible, or use slug as name
                # For simplicity, assume tag_name_slug is what we use for get_or_create_tag_info
                tag_name_for_prompt = tag_name_slug.replace('-', ' ').title()
                update_tag_with_article(tag_name_for_prompt, article) # This adds paper to tag's list

    # Now, regenerate descriptions and surveys for all known tags
    # The `update_tag_with_article` already triggers regeneration if a new paper is added.
    # If we want to force regeneration for all tags regardless of new papers (e.g., prompt changed):
    # for tag_slug in list(_GLOBAL_TAG_STORE.keys()): # list() for safe iteration if modified
    #     tag_name = _GLOBAL_TAG_STORE[tag_slug]['name']
    #     print(f"Force regenerating content for tag: {tag_name}")
    #     generate_tag_description(tag_name, force_regenerate=True)
    #     generate_tag_survey(tag_name, force_regenerate=True)
    print("--- Tag Update Process Complete ---")

