from typing import List, Optional
import os

from paper_reader.config import (
    ARTICLE_SUMMARY_VERBOSE,
    REBUILD_ALL,
    ENABLE_THINKING,
    DOCS_DIR,
    ENABLE_RAG_FOR_ARTICLES,
    EXTRACTED_MD_FILE,
    MODEL_INSTRUCT,
    MODEL_LONG,
    LOGGER,
    MAX_TOKENS_PER_ARTICLE_SUMMARY_PASS,
    MAX_TOKENS_TAGS,
    MAX_TOKENS_TLDR,
    SHORT_SUMMARIZED_MD_FILE,
    SUMMARIZED_MD_FILE,
    TLDR_MD_FILE,
)
from paper_reader.models import ArticleSummary, Content
from paper_reader.openai_utils import generate_completion, get_embedding
from paper_reader.prompts import (
    ARTICLE_SUMMARY_MERGE_PROMPT,
    ARTICLE_SUMMARY_SYSTEM,
    EXTRACT_TAGS_SYSTEM,
    ARTICLE_SHORT_SUMMARY_SYSTEM,
    TLDR_SUMMARY_SYSTEM,
    DEFAULT_SYSTEM_PROMPT,
    create_message_entry,
)
from paper_reader.utils import (
    load_text_and_embedding,
    save_text_and_embedding,
)
from paper_reader.vector_store import get_relevant_context_for_prompt
from paper_reader.tag_manager import prune_tags


async def _agenerate_and_save_content_tldr(
    text: str,
    output_dir: str,
    output_filename_md: str,
    max_tokens: int,
    rag_query_for_prompt: Optional[str] = None,  # Text to find relevant context for
    force_rebuild: bool = REBUILD_ALL,
) -> Optional[Content]:
    """Helper to generate, embed, and save a TLDR summary."""
    existing_content_obj = load_text_and_embedding(output_dir, output_filename_md)
    if (
        existing_content_obj and not force_rebuild
    ):
        return existing_content_obj

    LOGGER.info(f"Generating {output_filename_md} for {output_dir}...")

    # RAG: Augment prompt with relevant context if query is provided
    rag_ctx = ""
    if rag_query_for_prompt and ENABLE_RAG_FOR_ARTICLES:
        rag_ctx = get_relevant_context_for_prompt(rag_query_for_prompt, "articles")

    system_prompt = TLDR_SUMMARY_SYSTEM
    all_prompts: list = []
    all_prompts.append(
        create_message_entry(
            "user", "<-- This is the Article Summary -->\n\n{text}", text=text
        )
    )
    if rag_ctx and rag_ctx != "No existing content to retrieve from.":
        all_prompts.append(
            create_message_entry(
                "user", 
                f"<-- This is some context from similar papers -->\n\n{rag_ctx}"
            )
        )

    generated_text = await generate_completion(
        prompt=all_prompts,
        system_prompt=system_prompt,
        max_tokens=max_tokens,
        temperature=0.3,
    )

    if not generated_text:
        LOGGER.error(f"Failed to generate TLDR for {output_dir}")
        return None

    embedding_vector = get_embedding(generated_text)
    save_text_and_embedding(
        output_dir, output_filename_md, generated_text, embedding_vector
    )
    return Content(content=generated_text, vector=embedding_vector)


async def _agenerate_and_save_content_article_summary(
    text_to_process: str,
    output_dir: str,
    output_filename_md: str,
    previous_content_for_prompt: str = "N/A",
    force_rebuild: bool = REBUILD_ALL,
    thinking: bool = ENABLE_THINKING,
) -> Optional[Content]:
    """Helper to generate, embed, and save an article summary."""
    existing_content_obj = load_text_and_embedding(output_dir, output_filename_md)
    if existing_content_obj and not force_rebuild:
        return existing_content_obj

    LOGGER.info(f"Generating {output_filename_md} for {output_dir}...")

    # Use RAG to augment the prompt if enabled
    rag_context = ""
    if ENABLE_RAG_FOR_ARTICLES:
        # Query to find related articles for context
        rag_query = text_to_process[:500]  # Use start of the article for query
        rag_context = get_relevant_context_for_prompt(rag_query, "articles")
    
    # Create system prompt
    system_prompt = ARTICLE_SUMMARY_SYSTEM
    
    # Build the prompt
    all_prompts = []
    all_prompts.append(
        create_message_entry(
            "user", 
            f"<-- This is the paper content -->\n\n{text_to_process}"
        )
    )

    # Add RAG context if available
    if rag_context and rag_context != "No existing content to retrieve from.":
        all_prompts.append(
            create_message_entry(
                "user",
                f"<-- This is some context from similar papers -->\n\n{rag_context}"
            )
        )

    # Generate the summary
    generated_text = await generate_completion(
        prompt=all_prompts,
        system_prompt=system_prompt,
        model=MODEL_LONG,
        max_tokens=MAX_TOKENS_PER_ARTICLE_SUMMARY_PASS,
        temperature=0.3,  # Lower temperature for more accurate summarization
        thinking=thinking,
    )

    if not generated_text:
        LOGGER.error(f"Failed to generate summary for {output_dir}")
        return None
    
    # Handle case where output is too large - chunk it into parts and process
    if ARTICLE_SUMMARY_VERBOSE and len(generated_text.split()) > 1000:
        LOGGER.info("Summary is large, merging parts...")
        merged_text = await generate_completion(
            prompt=ARTICLE_SUMMARY_MERGE_PROMPT.format(parts=generated_text),
            system_prompt=DEFAULT_SYSTEM_PROMPT,
            model=MODEL_LONG,
            max_tokens=MAX_TOKENS_PER_ARTICLE_SUMMARY_PASS,
            temperature=0.3,
        )
        if merged_text:
            generated_text = merged_text

    # Create embedding for the summary
    embedding_vector = get_embedding(generated_text)
    
    # Save the summary
    save_text_and_embedding(
        output_dir, output_filename_md, generated_text, embedding_vector
    )
    
    return Content(content=generated_text, vector=embedding_vector)

async def _agenerate_and_save_content_short_summary(
    full_text: str,
    summary_text: str,
    output_dir: str,
    output_filename_md: str,
    force_rebuild: bool = REBUILD_ALL,
    thinking: bool = ENABLE_THINKING,
) -> Optional[Content]:
    """Generates a shorter, more concise summary of an article."""
    existing_content_obj = load_text_and_embedding(output_dir, output_filename_md)
    if existing_content_obj and not force_rebuild:
        return existing_content_obj

    LOGGER.info(f"Generating short summary for {output_dir}...")
    
    # Build the prompt
    all_prompts = []
    all_prompts.append(
        create_message_entry(
            "user", 
            f"<-- This is the paper content -->\n\n{full_text[:1000]}..."
        )
    )
    all_prompts.append(
        create_message_entry(
            "user", 
            f"<-- This is the detailed summary of the paper -->\n\n{summary_text}"
        )
    )
    all_prompts.append(
        create_message_entry(
            "user", 
            "Create a short summary (250-300 words) that captures the key points of this paper. "
            "Focus on the problem addressed, the proposed approach, and the significance of the results."
        )
    )

    # Generate the short summary
    generated_text = await generate_completion(
        prompt=all_prompts,
        system_prompt=ARTICLE_SHORT_SUMMARY_SYSTEM,
        model=MODEL_INSTRUCT,
        max_tokens=500,  # Appropriate length for short summary
        temperature=0.3,
        thinking=thinking,
    )

    if not generated_text:
        LOGGER.error(f"Failed to generate short summary for {output_dir}")
        return None
    
    # Create embedding for the summary
    embedding_vector = get_embedding(generated_text)
    
    # Save the short summary
    save_text_and_embedding(
        output_dir, output_filename_md, generated_text, embedding_vector
    )
    
    return Content(content=generated_text, vector=embedding_vector)

async def agenerate_tags(
    text: str,
    previous_tags: Optional[List[str]] = None,
    thinking: bool = ENABLE_THINKING,
) -> List[str]:
    """Generates tags for an article using NLP and returns a list of pruned tags."""
    if not text.strip():
        LOGGER.warning("Cannot generate tags for empty text")
        return []
    
    prompt = []
    prompt.append(
        create_message_entry(
            "user",
            f"<-- This is the paper content -->\n\n{text[:3000]}...\n\n"
            "Extract relevant academic tags from this paper. Focus on research areas, "
            "methods, and key concepts. Return a comma-separated list of tags only, no explanations."
        )
    )
    
    # If we have previous tags, we can use them as a starting point
    if previous_tags and len(previous_tags) > 0:
        tags_str = ", ".join(previous_tags)
        prompt.append(
            create_message_entry(
                "user",
                f"Here are some previously extracted tags: {tags_str}\n"
                "You can use these as a starting point, but feel free to add, remove, or modify tags."
            )
        )
    
    # Generate the tags
    tags_text = await generate_completion(
        prompt=prompt,
        system_prompt=EXTRACT_TAGS_SYSTEM,
        model=MODEL_INSTRUCT,
        max_tokens=MAX_TOKENS_TAGS,
        temperature=0.3,  # Lower temperature for more consistent tags
        thinking=thinking,
    )
    
    if not tags_text:
        LOGGER.warning("Failed to generate tags")
        return []
    
    # Parse the tags
    raw_tags = [tag.strip() for tag in tags_text.split(",") if tag.strip()]
    
    # Prune the tags to get a standardized list
    pruned_tags = await prune_tags(raw_tags)
    
    return pruned_tags


async def aprocess_article(
    paper_directory_name: str,
    article_title: str,
) -> Optional[ArticleSummary]:
    """Processes an article by generating summaries, TLDR, and tags."""
    LOGGER.info(f"Processing article: {article_title}")
    
    # Set up paths
    paper_dir = os.path.join(DOCS_DIR, paper_directory_name)
    if not os.path.exists(paper_dir) or not os.path.isdir(paper_dir):
        LOGGER.error(f"Paper directory not found: {paper_dir}")
        return None
    
    # Check if extracted.md exists
    extracted_path = os.path.join(paper_dir, EXTRACTED_MD_FILE)
    if not os.path.exists(extracted_path):
        LOGGER.error(f"Extracted content not found: {extracted_path}")
        return None
    
    # Load extracted content
    extracted_content = load_text_and_embedding(paper_dir, EXTRACTED_MD_FILE)
    if not extracted_content:
        LOGGER.error(f"Failed to load extracted content: {extracted_path}")
        return None
    
    # Initialize article summary object
    article_summary = ArticleSummary(
        title=article_title,
        paper_slug=paper_directory_name,
        paper_path=extracted_path,
        content=extracted_content,
        summary=None,
        short_summary=None,
        tldr=None,
        tags=[],
    )
    
    # Generate full summary
    LOGGER.info(f"Generating full summary for {article_title}")
    article_summary["summary"] = await _agenerate_and_save_content_article_summary(
        extracted_content["content"],
        paper_dir,
        SUMMARIZED_MD_FILE,
    )
    
    # If summary generation failed, return partial results
    if not article_summary["summary"]:
        LOGGER.error(f"Failed to generate summary for {article_title}")
        return article_summary
    
    # Generate short summary
    LOGGER.info(f"Generating short summary for {article_title}")
    article_summary["short_summary"] = await _agenerate_and_save_content_short_summary(
        extracted_content["content"],
        article_summary["summary"]["content"],
        paper_dir,
        SHORT_SUMMARIZED_MD_FILE,
    )
    
    # Generate TLDR
    LOGGER.info(f"Generating TLDR for {article_title}")
    article_summary["tldr"] = await _agenerate_and_save_content_tldr(
        article_summary["summary"]["content"],
        paper_dir,
        TLDR_MD_FILE,
        MAX_TOKENS_TLDR,
        rag_query_for_prompt=article_title,  # Use title as RAG query
    )
    
    # Generate tags
    LOGGER.info(f"Generating tags for {article_title}")
    article_summary["tags"] = await agenerate_tags(
        article_summary["summary"]["content"],
    )
    
    LOGGER.info(f"Finished processing article: {article_title}")
    LOGGER.info(f"Tags: {', '.join(article_summary['tags'])}")
    
    return article_summary
