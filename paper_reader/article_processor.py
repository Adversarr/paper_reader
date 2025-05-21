# article_processor.py
import os
from typing import List, Optional

from paper_reader.config import (
    ARTICLE_SUMMARY_VERBOSE,
    DEFAULT_REBUILD,
    DEFAULT_THINKING,
    DOCS_DIR,
    ENABLE_RAG_FOR_ARTICLES,
    EXTRACTED_MD_FILE,
    GPT_MODEL_FAST,
    LOGGER,
    MAX_TOKENS_PER_ARTICLE_SUMMARY_PASS,
    MAX_TOKENS_TLDR,
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
    TLDR_SUMMARY_SYSTEM,
    create_message_entry,
)
from paper_reader.utils import (
    ensure_dir_exists,
    load_text_and_embedding,
    save_text_and_embedding,
    slugify,
)
from paper_reader.vector_store import (
    get_relevant_context_for_prompt,
)  # For potential RAG in summarization


def _generate_and_save_content_tldr(
    text: str,
    output_dir: str,
    output_filename_md: str,
    max_tokens: int,
    rag_query_for_prompt: Optional[str] = None,  # Text to find relevant context for
    force_rebuild: bool = DEFAULT_REBUILD,
) -> Optional[Content]:
    """Helper to generate, embed, and save a piece of content."""
    existing_content_obj = load_text_and_embedding(output_dir, output_filename_md)
    if existing_content_obj and not force_rebuild:  # Check if we can reuse existing content
        print(f"Found existing {output_filename_md} in {output_dir}, loading.")
        return existing_content_obj

    print(f"Generating {output_filename_md} for {output_dir}...")

    # RAG: Augment prompt with relevant context if query is provided
    rag_ctx = ""
    if rag_query_for_prompt:
        # Example: get context from other article summaries
        rag_ctx = get_relevant_context_for_prompt(rag_query_for_prompt, source_type="articles", top_k=2)

    system_prompt = TLDR_SUMMARY_SYSTEM
    all_prompts: list = []
    all_prompts.append(create_message_entry("user", "<-- This is the Article Summary -->\n\n{text}", text=text))
    if rag_ctx:
        all_prompts.append(create_message_entry("user", "<-- This is the RAG Context -->\n\n{text}", text=rag_ctx))

    generated_text = generate_completion(
        prompt=all_prompts,
        system_prompt=system_prompt,
        max_tokens=max_tokens,
    )

    if not generated_text:
        print(f"Failed to generate {output_filename_md}.")
        return None

    embedding_vector = get_embedding(generated_text)
    save_text_and_embedding(output_dir, output_filename_md, generated_text, embedding_vector)
    return Content(content=generated_text, vector=embedding_vector)


def _generate_and_save_content_article_summary(
    text_to_process: str,
    output_dir: str,
    output_filename_md: str,
    previous_content_for_prompt: str = "N/A",
    force_rebuild: bool = DEFAULT_REBUILD,
    thinking: bool = DEFAULT_THINKING,
) -> Optional[Content]:
    """Helper to generate, embed, and save a piece of content."""

    _SUMMARY_STAGES = {
        # Explict Chain of Thought.
        "prelogue": "Generate the introduction part of the paper summary. Fill in the blanks(square bracket). Template: \n\n{text}",
        "pass1": "Generate your first pass reading summary of the paper. Fill in the blanks(square bracket). Template:\n\n{text}",
        "pass2": "Generate your second pass reading summary of the paper. Fill in the blanks(square bracket). Template:\n\n{text}",
        "pass3": "Generate your third pass reading summary of the paper. Fill in the blanks(square bracket). Template:\n\n{text}",
        "epilogue": "Generate the conclusion part of the paper summary. Fill in the blanks(square bracket). Template: \n\n{text}",
    }

    existing_content_obj = load_text_and_embedding(output_dir, output_filename_md)
    if existing_content_obj and not force_rebuild:  # Check if we can reuse existing content
        print(f"Found existing {output_filename_md} in {output_dir}, loading.")
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

    generated_text = generate_completion(
        prompt=all_prompts,
        max_tokens=MAX_TOKENS_PER_ARTICLE_SUMMARY_PASS,
        thinking=thinking,
    )

    if ARTICLE_SUMMARY_VERBOSE:
        print("#" * 50 + f"\nGenerated prelogue:\n{generated_text}\n" + "#" * 50)

    if not generated_text:
        LOGGER.error(f"Failed to generate {output_filename_md}.")
        return None

    all_prompts.append(create_message_entry("assistant", generated_text))
    all_generated.append(create_message_entry("assistant", generated_text))

    ####################### generate RAG context using the prompt
    # RAG: Augment prompt with relevant context if query is provided
    if ENABLE_RAG_FOR_ARTICLES:
        # Example: get context from other article summaries
        rag_context = get_relevant_context_for_prompt(generated_text, source_type="articles")
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

        generated_text = generate_completion(
            all_prompts,
            max_tokens=MAX_TOKENS_PER_ARTICLE_SUMMARY_PASS,
            thinking=thinking,
        )
        if not generated_text:
            print(f"Failed to generate {output_filename_md}.")
            return None
        # Append to chat history.
        all_prompts.append(create_message_entry("assistant", generated_text))
        all_generated.append(create_message_entry("assistant", generated_text))

        if ARTICLE_SUMMARY_VERBOSE:
            print("#" * 50 + f"\nGenerated {stage}:\n{generated_text}\n" + "#" * 50)

    ####################### generate the final summary
    merge_prompt = create_message_entry(role="user", template=ARTICLE_SUMMARY_MERGE_PROMPT)
    all_generated.append(merge_prompt)

    LOGGER.info("=> Merge summaries")
    generated_text = generate_completion(
        all_generated,
        ARTICLE_SUMMARY_SYSTEM,
        model=GPT_MODEL_FAST,
        thinking=thinking,
    )
    if generated_text is None:
        print(f"Failed to generate merged summary for {output_filename_md}.")
        return None

    ####################### save the final summary
    embedding_vector = get_embedding(generated_text)
    save_text_and_embedding(output_dir, output_filename_md, generated_text, embedding_vector)
    LOGGER.info(f'Done. Saved to "{output_dir}/{output_filename_md}"')
    if ARTICLE_SUMMARY_VERBOSE:
        print("#" * 50 + "Generated Summary Full:\n" + generated_text + "\n" + "#" * 50)
    return Content(content=generated_text, vector=embedding_vector)


def process_article(paper_directory_name: str, article_title: str) -> Optional[ArticleSummary]:
    """
    Processes a single article: loads extracted content, generates summaries,
    TLDR, section summaries, and extracts tags. Saves all outputs.
    `paper_directory_name` is the actual directory name on disk (e.g., "paper1").
    `article_title` is the human-readable title used for slugification if needed.
    """
    print(f"\nProcessing article: {article_title} in directory {paper_directory_name}")
    paper_slug = slugify(article_title)  # This should match paper_directory_name if generated by this system
    paper_path = os.path.join(DOCS_DIR, paper_directory_name)  # Use the provided directory name

    ensure_dir_exists(paper_path)

    # 1. Load original extracted content
    extracted_content_obj = load_text_and_embedding(paper_path, EXTRACTED_MD_FILE)
    if not extracted_content_obj:
        # Try to load just the MD if NPZ is missing, and then embed it
        md_path_only = os.path.join(paper_path, EXTRACTED_MD_FILE)
        if os.path.exists(md_path_only):
            with open(md_path_only, "r", encoding="utf-8") as f:
                raw_text = f.read()
            print(f"Found {EXTRACTED_MD_FILE}, generating its embedding...")
            embedding_vector = get_embedding(raw_text)
            save_text_and_embedding(paper_path, EXTRACTED_MD_FILE, raw_text, embedding_vector)  # This saves .npz
            extracted_content_obj = Content(content=raw_text, vector=embedding_vector)
        else:
            print(f"Error: {EXTRACTED_MD_FILE} not found in {paper_path}. Skipping article.")
            return None

    raw_text = extracted_content_obj["content"]

    # 2. Generate article-level summary
    # For iterative improvement, we could load an existing summary and pass it to the prompt.
    # Here, we just check if the file exists.
    existing_summary_obj = load_text_and_embedding(paper_path, SUMMARIZED_MD_FILE)
    previous_summary_text = existing_summary_obj["content"] if existing_summary_obj else "N/A"

    summary_obj = _generate_and_save_content_article_summary(
        text_to_process=raw_text,
        output_dir=paper_path,
        output_filename_md=SUMMARIZED_MD_FILE,
        previous_content_for_prompt=previous_summary_text,
    )
    if summary_obj is None:
        print(f"Error generating summary for {paper_path}. Skipping further processing.")
        return None

    # 3. Generate TLDR summary
    tldr_obj = _generate_and_save_content_tldr(
        text=summary_obj["content"],  # Use the generated summary for TLDR
        # text_to_process=raw_text,  # Or use summary_obj['content'] for TLDR from summary
        output_dir=paper_path,
        output_filename_md=TLDR_MD_FILE,
        max_tokens=MAX_TOKENS_TLDR,
    )

    # 4. Extract tags
    # Tags are usually extracted once. If re-running, we might want to update them.
    # For simplicity, we'll extract if not already present in a conceptual way.
    # A real system might store tags in the ArticleSummary file or a separate tags.json.
    # Here, we'll just generate them.
    print(f"Extracting tags for {paper_directory_name}...")
    tags_string = generate_completion(
        prompt=summary_obj["content"],
        system_prompt=EXTRACT_TAGS_SYSTEM,
        max_tokens=200,
    )
    tags_list: List[str] = []
    if tags_string:
        tags_list = [slugify(tag.strip()) for tag in tags_string.split(",") if tag.strip()]
        print(f"  Extracted tags: {tags_list}")
    else:
        print("  Failed to extract tags.")

    article_data = ArticleSummary(
        title=article_title,
        paper_slug=paper_slug,
        content_path=os.path.join(paper_path, EXTRACTED_MD_FILE),
        content=extracted_content_obj,
        summary=summary_obj,
        tldr=tldr_obj,
        tags=tags_list,
    )

    # Persist ArticleSummary object? For now, components are saved separately.
    # One could save this TypedDict as a JSON file in paper_path for easy loading.
    # e.g., with open(os.path.join(paper_path, "article_summary_meta.json"), "w") as f:
    #    json.dump(article_data, f, indent=2, cls=NumpyEncoder) # Needs a NumpyEncoder for ndarray

    return article_data
