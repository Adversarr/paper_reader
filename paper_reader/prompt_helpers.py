from pathlib import Path
from typing import Dict, List
import re
from paper_reader.config import LOGGER

LOADED_CACHE: Dict[str, str] = {}
PROMPT_INCLUDE_PATTERN = r"<!-- INCLUDE: ([^>]+) -->"


def _preprocess_includes(text: str, searched: List[str] = []) -> str:
    # 1. Find the first match
    match = re.search(PROMPT_INCLUDE_PATTERN, text)
    if not match:  # ok.
        return text

    # 2. Get the file name
    file_name = match.group(1).strip()
    if file_name in searched:
        raise ValueError(
            f"Circular reference detected: {file_name} is already being processed."
        )

    searched.append(file_name)
    included_content = load_prompt(file_name)
    included_content = _preprocess_includes(included_content, searched)
    searched.pop()

    # 3. Replace the include statement with the content
    front = text[: match.start()].strip()
    middle = included_content.strip()
    back = text[match.end() :].strip()
    text = f"{front}\n{middle}\n{back}"
    return _preprocess_includes(text, searched)  # Recursively check for more includes


def _load_content_actual(path: Path) -> str:
    raw_content = open(path, "r", encoding="utf-8").read()
    preprocessed_content = _preprocess_includes(raw_content, [])
    LOGGER.info(f"Loaded prompt from {str(path)}")
    return preprocessed_content


def load_prompt(path: str | Path) -> str:
    path = Path(path)
    str_path = str(path)  # Convert Path object to string for caching purposes
    if str_path in LOADED_CACHE:
        return LOADED_CACHE[str_path]

    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"The file at {path} does not exist, or is not a file.")

    content = _load_content_actual(path)
    LOADED_CACHE[str_path] = content  # Cache the content
    return content

# Test using prompts/article_summary.md
if __name__ == "__main__":
    test_path = "prompts/article_summary.md"
    content = load_prompt(test_path)
    content = load_prompt(test_path)
    print(f'Length of loaded content: {len(content)}')