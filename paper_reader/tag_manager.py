# tag_manager.py
import asyncio
from json import dumps
import os
from typing import Iterable, List, Dict, Optional
from paper_reader.models import TagInfo, ArticleSummary, Content
from paper_reader.prompt_helpers import load_prompt
from paper_reader.prompts import (
    TAG_PRUNE_SYSTEM,
    TAG_SURVEY_SYSTEM,
    TAG_UPDATE_SYSTEM,
    create_message_entry,
)
from paper_reader.utils import (
    ensure_dir_exists,
    human_readable_slugify,
    slugify,
    save_text_and_embedding,
    load_text_and_embedding,
)
from paper_reader.config import (
    DEFAULT_REBUILD_TAGS,
    MODEL_DEFAULT,
    MODEL_FAST,
    MAX_TOKENS_TAG_ARTICLE,
    MODEL_INSTRUCT,
    TAGS_DIR,
    TAG_DESCRIPTION_MD_FILE,
    TAG_SURVEY_MD_FILE,
    MAX_TOKENS_TAG_DESCRIPTION,
    DOCS_DIR,
    SUMMARIZED_MD_FILE,
    LOGGER,
)
from paper_reader.openai_utils import (
    generate_completion,
    get_embedding,
    # generate_completion,
)

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
        return _GLOBAL_TAG_STORE[tag_slug]  # Return in-memory if already loaded

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
    if tag_slug in _GLOBAL_TAG_STORE:  # If we have some in-memory state
        related_papers = _GLOBAL_TAG_STORE[tag_slug].get("related_paper_slugs", [])

    tag_info = TagInfo(
        name=tag_name,
        tag_slug=tag_slug,
        description=description_content,
        survey=survey_content,
        related_paper_slugs=related_papers,  # This will be updated
    )
    _GLOBAL_TAG_STORE[tag_slug] = tag_info  # Update global store
    return tag_info


async def aprocess_tag(tag_info: TagInfo):
    """Generates the description and survey for a tag."""
    tag_name = tag_info["name"]
    tag_slug = tag_info["tag_slug"]
    tag_dir = os.path.join(TAGS_DIR, tag_slug)

    ensure_dir_exists(tag_dir)  # Ensure the directory exists

    # Generate description and survey for the tag
    await agenerate_tag_survey(tag_name, True)


async def agenerate_tag_survey(tag_name: str, force_regenerate: bool = False):
    """Generates (or re-generates) a survey of related papers for a tag."""
    tag_info = get_or_create_tag_info(tag_name)
    tag_slug = tag_info["tag_slug"]
    tag_dir = os.path.join(TAGS_DIR, tag_slug)

    if not force_regenerate and tag_info["survey"] and tag_info["survey"]["content"]:
        LOGGER.info(f"Survey for tag '{tag_name}' already exists. Skipping generation.")
        return

    LOGGER.info(f"Start generating survey for tag: {tag_name}")

    system_prompt = TAG_SURVEY_SYSTEM
    ####################### 1. prelogue
    previous_survey_text = tag_info["survey"]["content"] if tag_info["survey"] else "N/A"

    # only select the first two. TODO: use ranking.
    selected_papers = tag_info.get("related_paper_slugs", [])[:2]
    related_papers_summaries = ""
    for paper_slug in selected_papers:
        paper_dir = os.path.join(DOCS_DIR, paper_slug)
        summary_content_obj = load_text_and_embedding(paper_dir, SUMMARIZED_MD_FILE)
        if summary_content_obj is None:
            LOGGER.error(f"Failed to load summary for paper: {paper_slug}")
            continue
        previous_survey_text += f"""
            <!-- Paper Summary for {paper_slug} -->

            {summary_content_obj['content']}
        """

    all_prompts = [
        create_message_entry(
            role="user",
            template="<!-- Previous Survey -->\n\n{previous_survey_text}\n\n<!-- End Previous Survey -->\n\n"
            "<!-- Related Papers Summaries -->\n\n{related_papers_summaries}\n\n",
            previous_survey_text=previous_survey_text,
            related_papers_summaries=related_papers_summaries,
        )
    ]

    all_prompts.append(
        create_message_entry(
            role="user",
            template="Generate the introduction part of the survey for {tag}. Fill in the blanks(square bracket]). "
            "\n\nYou do not need to generate the full text, Template: \n\n{text}",
            text=load_prompt("prompts/tag_survey/prelogue.md"),
            tag=tag_name,
        )
    )

    # TODO: include article's output.
    tag_introduction_part = await generate_completion(
        all_prompts,
        system_prompt=system_prompt,
        model=MODEL_DEFAULT,
        max_tokens=MAX_TOKENS_TAG_DESCRIPTION,
    )
    if tag_introduction_part is None:
        LOGGER.error(f"Failed to generate prelogue for tag survey: {tag_name}")
        return
    all_prompts.append(create_message_entry(role="assistant", template=tag_introduction_part))

    # Collect related articles info for context
    article_infos = []
    for paper_slug in tag_info.get("related_paper_slugs", []):
        paper_dir = os.path.join(DOCS_DIR, paper_slug)
        summary_content_obj = load_text_and_embedding(paper_dir, SUMMARIZED_MD_FILE)
        if summary_content_obj is None:
            LOGGER.error(f"Failed to load summary for paper: {paper_slug}")
            continue
        article_prompt = [
            create_message_entry(
                role="user",
                template="<!-- This is the paper summary -->\n\n{content}",
                content=summary_content_obj["content"],
            ),
            create_message_entry(
                role="user",
                template="Generate the 'Related Articles' part of the survey (just one item in this prompt)."
                "Fill in the blanks (the [square bracket]). Template: \n\n{text}",
                text=load_prompt("prompts/tag_survey/_article1.md"),
            ),
        ]
        LOGGER.info(f"Tag \"{tag_name}\" related paper: {paper_slug}")
        response = await generate_completion(
            all_prompts + article_prompt,
            system_prompt,
            model=MODEL_DEFAULT,
            max_tokens=MAX_TOKENS_TAG_ARTICLE,
        )
        if response is None:
            LOGGER.error(f"Failed to generate completion for paper: {paper_slug}")
            continue
        article_infos.append(create_message_entry("assistant", response))

    survey_text = await generate_completion(
        all_prompts
        + article_infos
        + [
            create_message_entry(
                role="user",
                template="Generate the final survey text based on the provided articles.",
            )
        ],
        system_prompt,
        temperature=0.2,
    )

    if survey_text:
        embedding_vector = get_embedding(survey_text)
        save_text_and_embedding(tag_dir, TAG_SURVEY_MD_FILE, survey_text, embedding_vector)
        tag_info["survey"] = Content(content=survey_text, vector=embedding_vector)
        _GLOBAL_TAG_STORE[tag_slug] = tag_info  # Update store
        LOGGER.info(f"  Generated and saved survey for tag '{tag_name}'.")
    else:
        LOGGER.info(f"  Failed to generate survey for tag '{tag_name}'.")


async def prune_tags(tags: Iterable[str]) -> List[str]:
    """
    Cleans and slugifies a list of tags.
    This function ensures that tags are unique and properly formatted.
    """
    tags_stripped = [tag.strip() for tag in tags]
    prompt_tags = ",".join(tags_stripped)
    pruned = await generate_completion(
        prompt=prompt_tags,
        system_prompt=TAG_PRUNE_SYSTEM,
        model=MODEL_FAST,
        thinking=False,
    )

    if pruned:
        LOGGER.info(f"Pruned tags: {pruned}")
        pruned_tags = [tag.strip() for tag in pruned.split(",") if tag.strip()]
        # Ensure uniqueness and slugify
        unique_tags = set(pruned_tags)
        return [slugify(tag) for tag in unique_tags]
    else:
        LOGGER.error(f"Failed to prune tags: {tags_stripped}")
        return [slugify(tag) for tag in tags_stripped]


async def update_tags(article: ArticleSummary, pruned_tags: List[str]):
    """
    Update the tags using pruned_tags
    """
    if article["tags"] is None:
        raise ValueError(f"Article '{article['title']}' has no tags. Cannot update.")
    pruned_tags_s = ",".join(pruned_tags)
    article_tags_s = ",".join(article["tags"])
    all_prompts = []

    if article["short_summary"] is not None:
        all_prompts.append(
            create_message_entry(
                role="user",
                template="<!-- Article Summary -->\n\n{summary}",
                summary=article["short_summary"]["content"],
            )
        )
    all_prompts += [
        create_message_entry(
            role="user",
            template="<!-- Pruned Tags -->\n\n{tags}",
            tags=pruned_tags_s,
        ),
        create_message_entry(
            role="user",
            template="<!-- Article Tags -->\n\n{tags}",
            tags=article_tags_s,
        ),
    ]

    output = await generate_completion(
        prompt=all_prompts,
        system_prompt=TAG_UPDATE_SYSTEM,
        model=MODEL_INSTRUCT,
        temperature=0.2,
    )
    if output is None:
        LOGGER.error(f"Failed to update tags for article '{article['title']}'. Tags: {article_tags_s}")
        return None

    return _str_to_list(output)


async def process_all_tags_iteratively(all_articles: List[ArticleSummary]):
    """
    Iteratively updates all tags based on the full list of articles.
    This function is called after all articles are processed.
    It ensures that tag descriptions and surveys are (re)generated
    with the full context of all related papers.
    """

    # 1. get all the tags
    all_tags = set()
    for article in all_articles:
        if article.get("tags"):
            all_tags.update(article["tags"])
        else:
            LOGGER.warning(f"Article '{article['title']}' has no tags. Skipping.")
    all_tags = list(all_tags)

    # 1.1 use LLM to fix duplicated tags.
    if DEFAULT_REBUILD_TAGS:
        all_tags = await prune_tags(all_tags)
        LOGGER.info(f"All tags after pruning: {all_tags}")
        tasks = []
        for article in all_articles:
            if article.get("tags"):
                tasks.append(update_tags(article, all_tags))
        pruned_tags_list = await asyncio.gather(*tasks)

        for article, pruned_tags in zip(all_articles, pruned_tags_list):
            existing_tags = article.get("tags")
            if existing_tags:
                pruned_tags.sort()
                existing_tags.sort()
                pruned_tags = await update_tags(article, all_tags)
                if pruned_tags is not None:
                    pruned_tags = [slugify(tag.strip()) for tag in pruned_tags]
                    article["tags"] = pruned_tags
                    # Update the tag file
                    tag_json_path = os.path.join(article["paper_path"], "tags.json")
                    set_diff = (set(existing_tags) - set(pruned_tags)) | (set(pruned_tags) - set(existing_tags))
                    if set_diff:
                        LOGGER.info(f"Updating tags for article '{article['title']}'."
                            f"\nold: {existing_tags}."
                            f"\nnew: {pruned_tags}.")
                        with open(tag_json_path, "w") as f:
                            f.write(dumps(article["tags"], indent=2))
                            LOGGER.info(f"Saved tags to {tag_json_path}")
                else:
                    LOGGER.error(f"Failed to update tags for article '{article['title']}'.")
        all_tags = []
        for article in all_articles:
            if article.get("tags"):
                all_tags.extend(article["tags"])
        all_tags = list(set(all_tags))

    # 1.2 prepare all the tags.
    for tag in all_tags:
        get_or_create_tag_info(human_readable_slugify(tag))
    for article in all_articles:
        if article.get("tags"):
            for tag_name_slug in article["tags"]:
                _GLOBAL_TAG_STORE[tag_name_slug]["related_paper_slugs"].append(article["paper_slug"])

    tasks = []
    for tag in all_tags:
        info = get_or_create_tag_info(tag)
        LOGGER.info(f'Tag "{info["name"]}" has {len(info["related_paper_slugs"])} related papers.')
        tasks.append(aprocess_tag(info))

    await asyncio.gather(*tasks)

    LOGGER.info("--- Tag Update Process Complete ---")
