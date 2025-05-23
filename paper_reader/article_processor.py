from typing import List, Optional
import os

from paper_reader.config import (
    ARTICLE_SUMMARY_TEMPERATURE,
    ARTICLE_SUMMARY_VERBOSE,
    MODEL_DEFAULT,
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
from paper_reader.prompt_helpers import load_prompt
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
    if existing_content_obj and not force_rebuild:
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
                "user", f"<-- This is some context from similar papers -->\n\n{rag_ctx}"
            )
        )

    generated_text = await generate_completion(
        prompt=all_prompts,
        system_prompt=system_prompt,
        max_tokens=max_tokens,
        temperature=ARTICLE_SUMMARY_TEMPERATURE,
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
    """Helper to generate, embed, and save a piece of content."""

    _SUMMARY_STAGES = {
        # Explict Chain of Thought.
        "prelogue": "Generate the introduction part of the summary. Fill in the blanks(square bracket). Do not include other contents. Template: \n\n{text}",
        "pass1": "Generate your first pass reading summary. Fill in the blanks(square bracket). Do not include other contents. Template:\n\n{text}",
        "pass2": "Generate your second pass reading summary. Fill in the blanks(square bracket). Do not include other contents. Template:\n\n{text}",
        "pass3": "Generate your third pass reading summary. Fill in the blanks(square bracket). Do not include other contents. Template:\n\n{text}",
        "epilogue": "Generate the conclusion part of the summary. Fill in the blanks(square bracket). Do not include other contents. Template: \n\n{text}",
    }

    existing_content_obj = load_text_and_embedding(output_dir, output_filename_md)
    # Check if we can reuse existing content
    if existing_content_obj and not force_rebuild:
        LOGGER.info(f"Found existing {output_filename_md} in {output_dir}, loading.")
        return existing_content_obj
    all_prompts = [
        create_message_entry(
            "user",
            "<!-- This is the Article Content -->\n\n{text}",
            text=text_to_process,
        ),
    ]

    if previous_content_for_prompt != "N/A":
        all_prompts.append(
            create_message_entry(
                "user",
                "<!-- This is the Previous Summary -->\n\n{text}",
                text=previous_content_for_prompt,
            )
        )

    ####################### generate the prelogue
    all_prompts.append(
        create_message_entry(
            "user",
            _SUMMARY_STAGES["prelogue"],
            text=load_prompt("prompts/article_summary/summary_prelogue.md"),
        )
    )
    LOGGER.info(f"Generating {output_filename_md} for {output_dir}...")

    # store all generated text
    all_generated = []

    generated_text = await generate_completion(
        prompt=all_prompts,
        system_prompt=ARTICLE_SUMMARY_SYSTEM,
        max_tokens=MAX_TOKENS_PER_ARTICLE_SUMMARY_PASS,
        thinking=thinking,
    )

    if ARTICLE_SUMMARY_VERBOSE:
        LOGGER.debug("#" * 50 + f"\nGenerated prelogue:\n{generated_text}\n" + "#" * 50)

    if not generated_text:
        LOGGER.error(f"Failed to generate {output_filename_md}.")
        return None

    all_prompts.append(create_message_entry("assistant", generated_text))
    all_generated.append(create_message_entry("assistant", generated_text))

    ####################### generate RAG context using the prompt
    # RAG: Augment prompt with relevant context if query is provided
    if ENABLE_RAG_FOR_ARTICLES:
        # Example: get context from other article summaries
        rag_context = get_relevant_context_for_prompt(generated_text, "articles")
        rag_context += "\n\n"
        rag_context += get_relevant_context_for_prompt(generated_text, "tags")
        all_prompts.append(create_message_entry(role="user", template=rag_context))

    ####################### generate all other stages.
    other_stages = ["pass1", "pass2", "pass3", "epilogue"]
    for stage in other_stages:
        # Instruction prompt
        stage_prompt = _SUMMARY_STAGES[stage]
        instruct_prompt = create_message_entry(
            "user",
            stage_prompt,
            text=load_prompt(f"prompts/article_summary/summary_{stage}.md"),
        )
        all_prompts.append(instruct_prompt)
        LOGGER.info(f"=> Generate {stage}")

        generated_text = await generate_completion(
            all_prompts,
            system_prompt=ARTICLE_SUMMARY_SYSTEM,
            max_tokens=MAX_TOKENS_PER_ARTICLE_SUMMARY_PASS,
            thinking=thinking,
        )
        if not generated_text:
            LOGGER.error(f"Failed to generate {output_filename_md}.")
            return None
        # Append to chat history.
        all_prompts.append(create_message_entry("assistant", generated_text))
        all_generated.append(create_message_entry("assistant", generated_text))

        if ARTICLE_SUMMARY_VERBOSE:
            LOGGER.debug(
                "#" * 50 + f"\nGenerated {stage}:\n{generated_text}\n" + "#" * 50
            )

    ####################### generate the final summary
    merge_prompt = create_message_entry("user", template=ARTICLE_SUMMARY_MERGE_PROMPT)
    all_generated.append(merge_prompt)

    LOGGER.info("=> Merge summaries")
    generated_text = await generate_completion(
        all_generated,
        ARTICLE_SUMMARY_SYSTEM,
        model=MODEL_INSTRUCT,
        thinking=False,
    )
    if generated_text is None:
        LOGGER.error(f"Failed to generate merged summary for {output_filename_md}.")
        return None

    ####################### save the final summary
    embedding_vector = get_embedding(generated_text)
    save_text_and_embedding(
        output_dir, output_filename_md, generated_text, embedding_vector
    )
    LOGGER.info(f'Done. Saved to "{output_dir}/{output_filename_md}"')
    if ARTICLE_SUMMARY_VERBOSE:
        LOGGER.debug(
            "#" * 50 + "Generated Summary Full:\n" + generated_text + "\n" + "#" * 50
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
            "user", f"<-- This is the paper content -->\n\n{full_text[:1000]}..."
        )
    )
    all_prompts.append(
        create_message_entry(
            "user",
            f"<-- This is the detailed summary of the paper -->\n\n{summary_text}",
        )
    )
    all_prompts.append(
        create_message_entry(
            "user",
            "Create a short summary (250-300 words) that captures the key points of this paper. "
            "Focus on the problem addressed, the proposed approach, and the significance of the results.",
        )
    )

    # Generate the short summary
    generated_text = await generate_completion(
        prompt=all_prompts,
        system_prompt=ARTICLE_SHORT_SUMMARY_SYSTEM,
        model=MODEL_INSTRUCT,
        max_tokens=500,  # Appropriate length for short summary
        temperature=ARTICLE_SUMMARY_TEMPERATURE,
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
            "methods, and key concepts. Return a comma-separated list of tags only, no explanations.",
        )
    )

    # If we have previous tags, we can use them as a starting point
    if previous_tags and len(previous_tags) > 0:
        tags_str = ", ".join(previous_tags)
        prompt.append(
            create_message_entry(
                "user",
                f"Here are some previously extracted tags: {tags_str}\n"
                "You can use these as a starting point, but feel free to add, remove, or modify tags.",
            )
        )

    # Generate the tags
    tags_text = await generate_completion(
        prompt=prompt,
        system_prompt=EXTRACT_TAGS_SYSTEM,
        model=MODEL_INSTRUCT,
        max_tokens=MAX_TOKENS_TAGS,
        temperature=ARTICLE_SUMMARY_TEMPERATURE,  # Lower temperature for more consistent tags
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
    if not article_summary["short_summary"]:
        LOGGER.error(f"Failed to generate short summary for {article_title}")
        return article_summary

    # Generate TLDR
    LOGGER.info(f"Generating TLDR for {article_title}")
    article_summary["tldr"] = await _agenerate_and_save_content_tldr(
        article_summary["short_summary"]["content"],
        paper_dir,
        TLDR_MD_FILE,
        MAX_TOKENS_TLDR,
        rag_query_for_prompt=article_title,  # Use title as RAG query
    )

    # Generate tags
    LOGGER.info(f"Generating tags for {article_title}")
    article_summary["tags"] = await agenerate_tags(
        article_summary["short_summary"]["content"],
    )

    LOGGER.info(f"Finished processing article: {article_title}")
    LOGGER.info(f"Tags: {', '.join(article_summary['tags'])}")

    return article_summary
