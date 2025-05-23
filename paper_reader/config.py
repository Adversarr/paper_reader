# config.py
import os
import sys
from typing import Any

from dotenv import load_dotenv
from loguru import logger as LOGGER  # noqa: F401

_custom_format = (
    "<green>{time: HH:mm:ss}</green> | "
    "<level>{level: <5}</level> | "
    "<level>{message}</level>"
)
LOGGER.remove()
LOGGER.add(sink=sys.stdout, format=_custom_format, level="DEBUG" if os.getenv("DEBUG") else "INFO")

load_dotenv()

# --- Paths ---
VAULT_DIR = "vault"
DOCS_DIR = os.path.join(VAULT_DIR, "docs")
TAGS_DIR = os.path.join(VAULT_DIR, "tags")

# --- OpenAI API Configuration ---
# Make sure to set your OPENAI_API_KEY environment variable
PROVIDER: str = os.getenv("PROVIDER", "bailian")  # type: ignore
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", PROVIDER)  # type: ignore

def _get_api_base(provider: str) -> str:
    """Return the API base URL based on the provider name."""
    if provider == "openai":
        return "https://api.openai.com/v1"
    elif provider == "bailian":
        return "https://dashscope.aliyuncs.com/compatible-mode/v1"
    elif provider == "azure":
        return os.getenv("AZURE_API_BASE", "")
    elif provider == "minimax":
        return "https://api.minimax.chat/v1"
    elif provider == "anthropic":
        return "https://api.anthropic.com/v1"
    elif provider == "ollama":
        return os.getenv("OLLAMA_API_BASE", "http://localhost:11434/api")
    elif provider == 'volcengine':
        return "https://ark.cn-beijing.volces.com/api/v3"
    elif provider == "openrouter":
        return "https://api.openrouter.ai/v1"
    elif provider == 'siliconflow':
        return "https://api.siliconflow.cn/v1/chat/completions"
    elif provider == "custom":
        custom_base = os.getenv("CUSTOM_API_BASE", "")
        if not custom_base:
            raise ValueError("CUSTOM_API_BASE environment variable not set for custom provider")
        return custom_base
    else:
        raise ValueError(f"Unsupported provider: {provider}")


BASE_URL = _get_api_base(PROVIDER)
EMBEDDING_BASE_URL = _get_api_base(EMBEDDING_PROVIDER)
OPENAI_API_KEY = os.getenv("API_KEY", "")  # Ensure this is set in your environment variables
if not OPENAI_API_KEY:
    raise ValueError("API_KEY environment variable must be set.")
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY", OPENAI_API_KEY)  # Ensure this is set in your environment variables

# --- Model Configuration ---

# NOTE: These model names are suitable only for bailian api.
MODEL_DEFAULT = os.getenv("MODEL_DEFAULT", "qwen-plus")
MODEL_FAST = os.getenv("MODEL_FAST", MODEL_DEFAULT)
MODEL_TAG = os.getenv("MODEL_TAG", MODEL_DEFAULT)
MODEL_LONG = os.getenv("MODEL_LONG", MODEL_DEFAULT)
MODEL_INSTRUCT = os.getenv("MODEL_INSTRUCT", MODEL_DEFAULT)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-v3")
ARTICLE_SUMMARY_TEMPERATURE = float(os.getenv("ARTICLE_SUMMARY_TEMPERATURE", "0.7"))


NO_THINK_EXTRA_BODY: dict[str, Any] = {"enable_thinking": False}
THINK_EXTRA_BODY: dict[str, Any] = {"enable_thinking": True}

# --- File Names ---
EXTRACTED_MD_FILE = "extracted.md"
SUMMARIZED_MD_FILE = "summarized.md"
SHORT_SUMMARIZED_MD_FILE = "short_summarized.md"
TLDR_MD_FILE = "tldr.md"
TAGS_JSON_FILE = "tags.json"
TAG_DESCRIPTION_MD_FILE = "description.md"
TAG_SURVEY_MD_FILE = "survey.md"

# --- Other Constants ---
SECTION_SEPARATOR = os.getenv("SECTION_SEPARATOR", "")  # <!-- SEPARATOR -->
DEFAULT_MAX_TOKENS = int(os.getenv("DEFAULT_MAX_TOKENS", "16384"))
DEFAULT_TEMPERATURE = float(os.getenv("DEFAULT_TEMPERATURE", "0.7"))
REBUILD_ALL = os.getenv("REBUILD_ALL", "True").lower() == "true"
REBUILD_ALL_TAGS = os.getenv("REBUILD_ALL_TAGS", str(REBUILD_ALL)).lower() == "true"
PRUNE_ALL_TAGS = os.getenv("PRUNE_ALL_TAGS", str(REBUILD_ALL)).lower() == "true"
ENABLE_THINKING = os.getenv("ENABLE_THINKING", "False").lower() == "true"
DEFAULT_STREAM = os.getenv("DEFAULT_STREAM", "True").lower() == "true"
TAG_SURVEY_THRESHOLD = int(os.getenv("TAG_SURVEY_THRESHOLD", "2")) # at least 2 articles

ARTICLE_SUMMARY_VERBOSE = os.getenv("ARTICLE_SUMMARY_VERBOSE", "False").lower() == "true"
# --- RAG Configuration ---
ENABLE_RAG_FOR_ARTICLES = os.getenv("ENABLE_RAG_FOR_ARTICLES", "True").lower() == "true"
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "2"))

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
MAX_CONCURRENT = int(os.getenv("MAX_CONCURRENT", "4"))


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
