# config.py
import os
from typing import Any
from dotenv import load_dotenv
from loguru import logger as LOGGER  # noqa: F401

load_dotenv()

# --- Paths ---
VAULT_DIR = "vault"
DOCS_DIR = os.path.join(VAULT_DIR, "docs")
TAGS_DIR = os.path.join(VAULT_DIR, "tags")

# --- OpenAI API Configuration ---
# Make sure to set your OPENAI_API_KEY environment variable
PROVIDER: str = os.getenv("PROVIDER", "bailian")  # type: ignore


def _get_api_base(provider: str) -> str:
    """Return the API base URL for the given provider."""
    base_urls = {
        "openai": "https://api.openai.com/v1",
        "bailian": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "openrouter": "https://api.openrouter.ai/v1",
        "volcengine": "https://ark.cn-beijing.volces.com/api/v3",
    }
    if "BASE_URL" in os.environ:
        base_urls["custom"] = os.environ["BASE_URL"]  # Use custom base URL if provided
    provider = provider.lower()
    if provider not in base_urls:
        raise ValueError(f"Unknown provider: {provider}.")
    return base_urls[provider]


BASE_URL = _get_api_base(PROVIDER)
OPENAI_API_KEY = os.getenv("API_KEY", "")  # Ensure this is set in your environment variables
if not OPENAI_API_KEY:
    raise ValueError("API_KEY environment variable must be set.")
# --- Model Configuration ---

# NOTE: These model names are suitable only for bailian api.
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "qwen-turbo")
GPT_MODEL_DEFAULT = os.getenv("GPT_MODEL_DEFAULT", DEFAULT_MODEL)
GPT_MODEL_FAST = os.getenv("GPT_MODEL_FAST", DEFAULT_MODEL)
GPT_MODEL_TAG = os.getenv("GPT_MODEL_TAG", DEFAULT_MODEL)
GPT_MODEL_LONG = os.getenv("GPT_MODEL_LONG", DEFAULT_MODEL)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-v3")


NO_THINK_EXTRA_BODY: dict[str, Any] = {"enable_thinking": False}
THINK_EXTRA_BODY: dict[str, Any] = {"enable_thinking": True}

# --- File Names ---
EXTRACTED_MD_FILE = "extracted.md"
SUMMARIZED_MD_FILE = "summarized.md"
SECTIONS_SUMMARIZED_MD_FILE = "sections_summarized.md"
TLDR_MD_FILE = "tldr.md"
TAG_DESCRIPTION_MD_FILE = "description.md"
TAG_SURVEY_MD_FILE = "survey.md"

# --- Other Constants ---
SECTION_SEPARATOR = os.getenv("SECTION_SEPARATOR", "")  # <!-- SEPARATOR -->
DEFAULT_MAX_TOKENS = int(os.getenv("DEFAULT_MAX_TOKENS", "16384"))
DEFAULT_TEMPERATURE = float(os.getenv("DEFAULT_TEMPERATURE", "0.7"))
DEFAULT_REBUILD = os.getenv("DEFAULT_REBUILD", "True").lower() == "true"
DEFAULT_REBUILD_TAGS = os.getenv("DEFAULT_REBUILD_TAGS", str(DEFAULT_REBUILD)).lower() == "true"
DEFAULT_THINKING = os.getenv("DEFAULT_THINKING", "False").lower() == "true"
DEFAULT_STREAM = os.getenv("DEFAULT_STREAM", "True").lower() == "true"

ARTICLE_SUMMARY_VERBOSE = os.getenv("ARTICLE_SUMMARY_VERBOSE", "False").lower() == "true"
ENABLE_RAG_FOR_ARTICLES = os.getenv("ENABLE_RAG_FOR_ARTICLES", "True").lower() == "true"

MAX_TOKENS_SUMMARY = int(os.getenv("MAX_TOKENS_SUMMARY", str(DEFAULT_MAX_TOKENS)))
MAX_TOKENS_TAG_SURVEY = int(os.getenv("MAX_TOKENS_TAG_SURVEY", str(MAX_TOKENS_SUMMARY)))
MAX_TOKENS_PER_ARTICLE_SUMMARY_PASS = int(
    os.getenv("MAX_TOKENS_PER_ARTICLE_SUMMARY_PASS", str(int(0.5 * DEFAULT_MAX_TOKENS)))
)
MAX_TOKENS_TLDR = int(os.getenv("MAX_TOKENS_TLDR", "300"))
MAX_TOKENS_TAGS = int(os.getenv("MAX_TOKENS_TAGS", "300"))
MAX_TOKENS_TAG_DESCRIPTION = int(os.getenv("MAX_TOKENS_TAG_DESCRIPTION", "1024"))
MAX_TOKENS_TAG_ARTICLE = int(os.getenv("MAX_TOKENS_TAG_ARTICLE", "1024"))

# -1 means no async, 0 means unlimited, >0 means limited
MAX_CONCURRENT = int(os.getenv("MAX_CONCURRENT", "2"))


# --- Prompts ---
# Using """...""" for multiline prompts

DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."

PROMPT_TAG_DESCRIPTION = """
Generate a concise, Wikipedia-like description for the tag: "{tag_name}".
The description should explain what this tag represents, its core concepts, and potentially its significance or common applications.
You can use the following related article summaries for context.

Related Article Summaries:
{related_article_summaries}

Previous Description (if any, for context and iterative improvement):
{previous_description}

Description for "{tag_name}":
"""
