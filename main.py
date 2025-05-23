# main.py
import os
from typing import List, Tuple
from article_processor import process_article
from tag_manager import update_tag_with_article, process_all_tags_iteratively
from models import ArticleSummary
from config import DOCS_DIR, VAULT_DIR, EXTRACTED_MD_FILE
from utils import ensure_dir_exists, slugify

def discover_papers() -> List[Tuple[str, str]]:
    """
    Discovers papers in the vault/docs directory.
    Assumes each paper is in a subdirectory (e.g., vault/docs/my_paper_title_slug/)
    and has an `extracted.md` file.
    Returns a list of tuples: (directory_name, paper_title_for_processing).
    The paper_title_for_processing can be derived from the directory name.
    """
    ensure_dir_exists(DOCS_DIR)
    papers_to_process: List[Tuple[str, str]] = []
    for item in os.listdir(DOCS_DIR):
        item_path = os.path.join(DOCS_DIR, item)
        if os.path.isdir(item_path):
            # Check if extracted.md exists
            if os.path.exists(os.path.join(item_path, EXTRACTED_MD_FILE)):
                # Use directory name as a base for title, can be improved
                # Assuming directory name is already a slug or a meaningful name
                # For title, we can de-slugify it or use it as is.
                # Let's assume the directory name itself is the `paper_directory_name`
                # and we derive a human-readable title from it for prompts.
                human_readable_title = item.replace('_', ' ').replace('-', ' ').title()
                papers_to_process.append((item, human_readable_title))
            else:
                print(f"Skipping directory {item_path}: {EXTRACTED_MD_FILE} not found.")
    return papers_to_process

def main():
    """
    Main function to orchestrate the processing of articles and tags.
    """
    print("Starting RAG System Processing...")
    ensure_dir_exists(VAULT_DIR)
    ensure_dir_exists(DOCS_DIR)
    ensure_dir_exists(os.path.join(VAULT_DIR, "tags")) # Ensure TAGS_DIR from config exists

    # --- Step 1: Discover and Process Articles ---
    # `papers` is a list of (directory_name, human_readable_title)
    papers_to_process_info: List[Tuple[str, str]] = discover_papers()
    
    if not papers_to_process_info:
        print("No papers found to process. Ensure `vault/docs/<paper_dir>/extracted.md` exists.")
        print("Example: vault/docs/my-first-paper/extracted.md")
        # Create a dummy paper for testing if none exist
        dummy_paper_dir_name = "example-paper-on-ai-ethics"
        dummy_paper_title = "Example Paper on AI Ethics"
        dummy_paper_path = os.path.join(DOCS_DIR, dummy_paper_dir_name)
        ensure_dir_exists(dummy_paper_path)
        dummy_extracted_md_path = os.path.join(dummy_paper_path, EXTRACTED_MD_FILE)
        if not os.path.exists(dummy_extracted_md_path):
            with open(dummy_extracted_md_path, "w", encoding='utf-8') as f:
                f.write(f"# {dummy_paper_title}\n\nThis is an example paper about AI ethics.\n\n"
                        f"<!-- SEPERATOR -->\n\nSection 2 discusses implications.\n\n"
                        f"<!-- SEPERATOR -->\n\nSection 3 concludes the findings.")
            print(f"Created a dummy paper for demonstration: {dummy_extracted_md_path}")
            papers_to_process_info.append((dummy_paper_dir_name, dummy_paper_title))


    processed_articles: List[ArticleSummary] = []
    for paper_dir_name, paper_title in papers_to_process_info:
        article_summary_data = process_article(paper_dir_name, paper_title)
        if article_summary_data:
            processed_articles.append(article_summary_data)
        else:
            print(f"Failed to process article in directory: {paper_dir_name}")

    # --- Step 2: Update Tags based on all processed articles ---
    # This step ensures tags are aware of all relevant articles before generating descriptions/surveys.
    # The `process_all_tags_iteratively` function will call `update_tag_with_article`
    # which in turn calls `generate_tag_description` and `generate_tag_survey`.
    if processed_articles:
        process_all_tags_iteratively(processed_articles)
    else:
        print("No articles were successfully processed. Skipping tag updates.")

    print("\nRAG System Processing Complete.")

if __name__ == "__main__":
    # Before running, ensure OPENAI_API_KEY environment variable is set.
    if not os.getenv("OPENAI_API_KEY") and False:
        print("Error: OPENAI_API_KEY environment variable is not set.")
        print("Please set it before running the script.")
    else:
        main()
