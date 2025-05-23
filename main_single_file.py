#!/usr/bin/env python3
# filepath: /home/adversarr/Repo/paper_reader/main_single_file.py
"""
Single-file utility to process an individual PDF file, extract its content and generate summaries.
This script leverages the paper_reader package functionality for a single-file workflow.
"""

import os
os.environ['MAX_COCONCURENCY'] = '1'  # Set maximum concurrency level
import sys
import argparse
import asyncio
import shutil
from pathlib import Path
from typing import Optional, Tuple

# Import from paper_reader package
from dotenv import load_dotenv
from paper_reader.config import (
    EXTRACTED_MD_FILE,
    SUMMARIZED_MD_FILE,
    SHORT_SUMMARIZED_MD_FILE,
    TLDR_MD_FILE,
    ENABLE_THINKING,
    LOGGER,
)
from extractor import PDFExtractor
from paper_reader.article_processor import (
    _agenerate_and_save_content_article_summary,
    _agenerate_and_save_content_short_summary,
    _agenerate_and_save_content_tldr
)
from paper_reader.utils import slugify, ensure_dir_exists


async def process_single_pdf(pdf_path: Path, output_dir: Optional[Path] = None) -> None:
    """
    Process a single PDF file, extracting content and generating summaries.
    
    Args:
        pdf_path: Path to the PDF file
        output_dir: Optional custom output directory (defaults to filename-based slug)
    """
    if not pdf_path.exists():
        LOGGER.error(f"PDF file not found: {pdf_path}")
        return
    
    LOGGER.info(f"Processing PDF: {pdf_path}")
    
    # Create extractor
    extractor = PDFExtractor()
    
    # Prepare temporary directory for extraction (simulating RAW_DIR)
    temp_dir = Path("temp_raw")
    ensure_dir_exists(str(temp_dir))
    temp_pdf_path = temp_dir / pdf_path.name
    shutil.copy(pdf_path, temp_pdf_path)
    
    try:
        # Extract content using PDFExtractor
        file_id = await extractor._upload_file(temp_pdf_path)
        LOGGER.info(f"Uploaded PDF, file ID: {file_id}")
        
        # Extract title
        title = await extractor._extract_title(file_id)
        LOGGER.info(f"Extracted title: {title}")
        
        # Determine output directory
        target_dir = output_dir if output_dir else Path(f"output/{slugify(title)}")
        ensure_dir_exists(str(target_dir))
        
        # Extract title and abstract
        title_and_abstract = await extractor._extract_title_and_abstract(file_id)
        
        # Extract last section
        last_section = await extractor._extract_last_section(file_id)
        
        # Extract main content
        await extractor._extract_main_content(
            file_id, 
            title_and_abstract, 
            last_section, 
            target_dir / EXTRACTED_MD_FILE
        )
        
        # Save a copy of the original PDF
        shutil.copy(pdf_path, target_dir / pdf_path.name)
        
        # Load extracted content for summarization
        extracted_path = target_dir / EXTRACTED_MD_FILE
        with open(extracted_path, "r", encoding="utf-8") as f:
            extracted_content = f.read()
        
        # Generate article summary
        await _agenerate_and_save_content_article_summary(
            text_to_process=extracted_content,
            output_dir=str(target_dir),
            output_filename_md=SUMMARIZED_MD_FILE,
            force_rebuild=True,
            thinking=ENABLE_THINKING
        )
        
        # Load generated summary for further processing
        with open(target_dir / SUMMARIZED_MD_FILE, "r", encoding="utf-8") as f:
            summary_content = f.read()
        
        # Generate short summary
        await _agenerate_and_save_content_short_summary(
            full_text=extracted_content,
            summary_text=summary_content,
            output_dir=str(target_dir),
            output_filename_md=SHORT_SUMMARIZED_MD_FILE,
            force_rebuild=True,
            thinking=ENABLE_THINKING
        )
        
        # Generate TLDR
        await _agenerate_and_save_content_tldr(
            text=summary_content,
            output_dir=str(target_dir),
            output_filename_md=TLDR_MD_FILE,
            max_tokens=200,
            force_rebuild=True
        )
        
        LOGGER.info(f"Completed processing {pdf_path}")
        LOGGER.info(f"Results saved to {target_dir}")
        
    except Exception as e:
        LOGGER.error(f"Error processing PDF: {e}")
    finally:
        # Clean up temporary files
        if temp_dir.exists():
            shutil.rmtree(temp_dir)


async def main():
    """Parse command line arguments and process the specified PDF."""
    load_dotenv()  # Load environment variables
    
    parser = argparse.ArgumentParser(description="Process a PDF file to extract content and generate summaries")
    parser.add_argument("pdf_path", type=str, help="Path to the PDF file to process")
    parser.add_argument("--output-dir", "-o", type=str, help="Custom output directory (optional)")
    
    args = parser.parse_args()
    
    # Validate PDF path
    pdf_path = Path(args.pdf_path)
    if not pdf_path.exists() or not pdf_path.is_file():
        LOGGER.error(f"PDF file not found: {pdf_path}")
        sys.exit(1)
    
    # Process PDF
    output_dir = Path(args.output_dir) if args.output_dir else None
    await process_single_pdf(pdf_path, output_dir)


if __name__ == "__main__":
    asyncio.run(main())