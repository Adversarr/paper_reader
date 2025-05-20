# config.py
import os
from dotenv import dotenv_values

# --- Paths ---
VAULT_DIR = "vault"
DOCS_DIR = os.path.join(VAULT_DIR, "docs")
TAGS_DIR = os.path.join(VAULT_DIR, "tags")

# --- OpenAI API Configuration ---
# Make sure to set your OPENAI_API_KEY environment variable
OPENAI_API_KEY = dotenv_values()['BAILIAN_APIKEY']
GPT_MODEL_SUMMARIZE = 'qwen-turbo-latest'
GPT_MODEL_TAG = 'qwen-turbo-latest'
EMBEDDING_MODEL = "text-embedding-v3"
OPENAI_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# GPT_MODEL_SUMMARIZE = "gpt-3.5-turbo-0125" # Good for summarization tasks
# GPT_MODEL_TAG = "gpt-3.5-turbo-0125"       # Good for extraction tasks
# EMBEDDING_MODEL = "text-embedding-3-small"

# --- File Names ---
EXTRACTED_MD_FILE = "extracted.md"
SUMMARIZED_MD_FILE = "summarized.md"
SECTIONS_SUMMARIZED_MD_FILE = "sections_summarized.md"
TLDR_MD_FILE = "tldr.md"
TAG_DESCRIPTION_MD_FILE = "description.md"
TAG_SURVEY_MD_FILE = "survey.md"

# --- Prompts ---
# Using """...""" for multiline prompts

PROMPT_ARTICLE_SUMMARY = """
Please provide a comprehensive summary of the following article.
Focus on its main contributions, methods, key findings, and potential implications.
The summary should be detailed enough to give a good understanding of the paper.

Article Content:
{text}

Previous Summary (if any, for context):
{previous_summary}

Summary:
"""

PROMPT_SECTION_SUMMARY = """
Summarize the following section of an article. Focus on the key points and arguments presented in this specific section.

Section Content:
{text}

Summary:
"""

PROMPT_TLDR_SUMMARY = """
Provide a very short "Too Long; Didn't Read" (TLDR) summary for the following article.
It should be 1-2 sentences and capture the absolute essence of the paper (around 50-100 words).

Article Content:
{text}

TLDR:
"""

PROMPT_EXTRACT_TAGS = """
Analyze the following article content and extract relevant keywords (tags).
These tags should include the problems addressed, core ideas/innovations, techniques/methods used, and any specific tools or frameworks mentioned.
Provide a comma-separated list of tags. Aim for 5-10 highly relevant tags.

Article Content:
{text}

Tags (comma-separated):
"""

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

PROMPT_TAG_SURVEY = """
Create a survey for the tag: "{tag_name}".
This survey should briefly introduce the topic represented by the tag and then list and summarize the key contributions of the following related articles.
Focus on how each article relates to the tag.

Tag Name: {tag_name}

Tag Description (for context):
{tag_description}

Related Articles (Title and Summary):
{related_articles_info}

Previous Survey (if any, for context and iterative improvement):
{previous_survey}

Survey for "{tag_name}":
"""

# --- Other Constants ---
SECTION_SEPARATOR = "<!-- SEPARATOR -->"
MAX_TOKENS_SUMMARY = 1024
MAX_TOKENS_TLDR = 150
MAX_TOKENS_TAG_DESCRIPTION = 500
MAX_TOKENS_TAG_SURVEY = 1500
