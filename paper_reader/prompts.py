from typing import Literal
from paper_reader.prompt_helpers import load_prompt

DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."
ARTICLE_SUMMARY_SYSTEM = load_prompt("prompts/article_summary.md")
TLDR_SUMMARY_SYSTEM = load_prompt("prompts/tldr_summary.md")
EXTRACT_TAGS_SYSTEM = load_prompt("prompts/extract_tags.md")
TAG_SURVEY_SYSTEM = load_prompt("prompts/tag_survey.md")
TAG_PRUNE_SYSTEM = load_prompt("prompts/prune_tags.md")
TAG_UPDATE_SYSTEM = load_prompt("prompts/update_tags.md")

ARTICLE_SUMMARY_MERGE_PROMPT = f"""
Provided are different parts of the full summary, base on this, merge and get the final result.

You should remove:
1. all the html comments, i.e. `<!-- xxx -->`
2. all the squared brackets (blanks), i.e. `[yyy]`

**Output Template:**

{load_prompt("prompts/article_summary/summary_full.md")}
"""


def create_message_entry(
    role: Literal["user", "assistant", "system"],
    template: str,
    **kwargs,
):
    if kwargs:
        content = template.format(**kwargs)
    else:
        content = template
    if role not in ["user", "assistant", "system"]:
        raise ValueError(f"Invalid role: {role}. Must be 'user', 'assistant', or 'system'.")
    return {"role": role, "content": content}


if __name__ == "__main__":
    print(len(DEFAULT_SYSTEM_PROMPT))
    print(len(ARTICLE_SUMMARY_SYSTEM))
    print(len(TLDR_SUMMARY_SYSTEM))
    print(len(EXTRACT_TAGS_SYSTEM))
    print(len(TAG_SURVEY_SYSTEM))
