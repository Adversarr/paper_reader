# main.py
import os
import asyncio
from typing import List, Tuple
from dotenv import load_dotenv
from paper_reader.article_processor import aprocess_article
from paper_reader.openai_utils import rich_print_consumption_table
from paper_reader.tag_manager import process_all_tags_iteratively
from paper_reader.config import DOCS_DIR, VAULT_DIR, EXTRACTED_MD_FILE, LOGGER
from paper_reader.utils import ensure_dir_exists


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
                # Convert slug back to a human-readable title for processing
                human_readable_title = item.replace("-", " ").replace("_", " ").title()
                papers_to_process.append((item, human_readable_title))
                LOGGER.info(f"Found paper: {human_readable_title} in {item}")
            else:
                LOGGER.warning(f"Directory {item} exists but missing {EXTRACTED_MD_FILE}")
    return papers_to_process


async def main():
    """
    Main function to orchestrate the processing of articles and tags.
    """

    global running
    running = True

    async def run_print_consumption():
        try:
            while running:
                await rich_print_consumption_table()
                await asyncio.sleep(10)  # Sleep for 5 seconds before checking again
        except asyncio.CancelledError:
            LOGGER.info("Consumption tracking task cancelled")

    print_consumption_task = asyncio.create_task(run_print_consumption())  # Schedule the task to run periodically

    try:
        LOGGER.info("Starting RAG System Processing...")
        load_dotenv()  # Load environment variables from .env file

        ensure_dir_exists(VAULT_DIR)
        ensure_dir_exists(DOCS_DIR)
        ensure_dir_exists(os.path.join(VAULT_DIR, "tags"))  # Ensure TAGS_DIR from config exists

        # --- Step 1: Discover and Process Articles ---
        # `papers` is a list of (directory_name, human_readable_title)
        papers_to_process_info: List[Tuple[str, str]] = discover_papers()

        if not papers_to_process_info:
            LOGGER.warning(
                "No papers found to process. Ensure `vault/docs/<paper_dir>/extracted.md` exists.\n"
                "Example: vault/docs/my-first-paper/extracted.md"
            )
            # Create a dummy paper for testing if none exist
            dummy_paper_dir_name = "example-paper-on-ai-ethics"
            dummy_paper_path = os.path.join(DOCS_DIR, dummy_paper_dir_name)
            ensure_dir_exists(dummy_paper_path)
            dummy_extracted_md_path = os.path.join(dummy_paper_path, EXTRACTED_MD_FILE)
            if not os.path.exists(dummy_extracted_md_path):
                with open(dummy_extracted_md_path, "w", encoding="utf-8") as f:
                    f.write("# Example Paper on AI Ethics\n\nThis is an example abstract for demonstration purposes only. It discusses ethical considerations in artificial intelligence deployment and governance frameworks.")

        tasks = []
        for paper_dir_name, paper_title in papers_to_process_info:
            tasks.append(aprocess_article(paper_dir_name, paper_title))
        processed_articles = await asyncio.gather(*tasks)

        # --- Step 2: Update Tags based on all processed articles ---
        # This step ensures tags are aware of all relevant articles before generating descriptions/surveys.
        # The `process_all_tags_iteratively` function will call `update_tag_with_article`
        # which in turn calls `generate_tag_description` and `generate_tag_survey`.
        if processed_articles:
            await process_all_tags_iteratively(processed_articles)
        else:
            LOGGER.error("No articles were successfully processed. Skipping tag updates.")

        LOGGER.info("\nRAG System Processing Complete.")
        await rich_print_consumption_table()
    finally:
        # Clean up background tasks
        running = False
        print_consumption_task.cancel()
        try:
            await print_consumption_task
        except asyncio.CancelledError:
            pass


if __name__ == "__main__":
    asyncio.run(main())
