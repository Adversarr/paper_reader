import asyncio
import os
import shutil
from pathlib import Path
from typing import List, Optional, Tuple

from paper_reader.config import MODEL_LONG, PROVIDER, VAULT_DIR, LOGGER
from paper_reader.openai_utils import aclient
from paper_reader.prompts import load_prompt
from paper_reader.utils import slugify


class PDFExtractor:
    """Handles PDF processing and content extraction."""

    def __init__(self):
        self.raw_dir = self._get_required_env("RAW_DIR")
        self.sys_prompt = load_prompt("prompts/pdf_extractor.md")
        self.temperature = float(os.getenv("EXTRACTOR_TEMPERATURE", "0.2"))  # Fixed typo
        self.model = os.getenv("EXTRACTOR_MODEL", MODEL_LONG)

    @staticmethod
    def _get_required_env(var_name: str) -> str:
        """Get required environment variable or raise error."""
        value = os.getenv(var_name)
        if not value:
            raise ValueError(f"Required environment variable {var_name} not set")
        return value

    @property
    def naming_prompt(self) -> str:
        """Prompt for extracting paper title."""
        return """
        What is the title of this paper? You should output the title only, no extra information or sentences.

        **Examples:**

        Attention Is All You Need

        A Survey on Deep Learning for Image Super-Resolution

        PhysGaussian: Physics-Integrated 3D Gaussians for Generative Dynamics
        """

    async def _upload_file(self, pdf_path: Path) -> str:
        """Upload PDF file and return file ID."""
        file_object = await aclient.files.create(
            file=pdf_path,
            purpose="file-extract", # type: ignore
        )
        return file_object.id

    async def _extract_title(self, file_id: str) -> str:
        """Extract paper title from PDF."""
        response = await aclient.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "system", "content": f"fileid://{file_id}"},
                {"role": "user", "content": self.naming_prompt},
            ],
            temperature=0.2,
            stream=False,
            n=1,
        )

        content = response.choices[0].message.content
        if not content:
            raise ValueError("Failed to extract title")
        return content.strip()

    async def _extract_title_and_abstract(self, file_id: str) -> str:
        """Extract title and abstract from PDF."""
        response = await aclient.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.sys_prompt},
                {"role": "system", "content": f"fileid://{file_id}"},
                {
                    "role": "user",
                    "content": "Give me the text content of the pdf. (just the title and abstract)",
                },
            ],
            stream=False,
            temperature=0.2,
        )

        content = response.choices[0].message.content
        if not content:
            raise ValueError("Failed to extract title and abstract")

        if response.usage:
            LOGGER.info(
                f"Usage: {response.usage.prompt_tokens} prompt tokens, {response.usage.completion_tokens} completion tokens"
            )

        return content.strip()

    async def _extract_last_section(self, file_id: str) -> str:
        """Extract last section/paragraph from PDF."""
        response = await aclient.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.sys_prompt},
                {"role": "system", "content": f"fileid://{file_id}"},
                {
                    "role": "user",
                    "content": "Give me the last paragraph or section of the pdf. (If appendix exists, it should be the appendix section, otherwise, its likely to be the conclusion section)",
                },
            ],
            stream=False,
            temperature=0.2,
        )

        content = response.choices[0].message.content
        if not content:
            raise ValueError("Failed to extract last section")

        if response.usage:
            LOGGER.info(f"Consumption: {response.usage.total_tokens} tokens")

        return content.strip()

    def _create_main_pages_prompt(self, title_and_abstract: str, last_section: str) -> str:
        """Create prompt for extracting main pages."""
        return f"""
        You are going to extract the main pages.

        I have extracted the following information:
        ```markdown
        Title and Abstract:
        {title_and_abstract}

        Last Paragraph or Section:
        {last_section}
        ```

        Please provide the main pages of the document:

        1. Your answer should begin with the title of the main pages.
        2. Your answer should end with the last paragraph or section I provided.
        3. You should begin with
        """

    async def _extract_main_content(
        self,
        file_id: str,
        title_and_abstract: str,
        last_section: str,
        output_file: Path,
    ) -> None:
        """Extract main content and stream to file."""
        prompt = self._create_main_pages_prompt(title_and_abstract, last_section)

        response = await aclient.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.sys_prompt},
                {"role": "system", "content": f"fileid://{file_id}"},
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
            stream=True,
            n=1,
        )

        written_chars = 0
        with open(output_file, "w", encoding="utf-8") as f:
            async for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    f.write(content)

                    last_written = written_chars
                    written_chars += len(content)

                    # Log progress every 1000 characters
                    if last_written // 1000 < written_chars // 1000:
                        LOGGER.info(f"Writing progress: {written_chars} chars to {output_file}")

    def _create_output_directory(self, slug: str) -> Path:
        """Create and return output directory path."""
        output_dir = Path(VAULT_DIR) / "docs" / slug
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    def _save_artifacts(self, output_dir: Path, pdf_path: Path, title: str) -> None:
        """Save PDF file and title to output directory."""
        shutil.copy(pdf_path, output_dir / "raw.pdf")

        with open(output_dir / "TITLE", "w", encoding="utf-8") as f:
            f.write(title)

    async def process_pdf(self, pdf_path: Path) -> Optional[Tuple[str, str]]:
        """Process a single PDF file."""
        try:
            pdf_name = pdf_path.stem
            LOGGER.info(f"Processing: {pdf_name}")

            # Upload file
            file_id = await self._upload_file(pdf_path)
            LOGGER.info(f"{pdf_name} -> {file_id}")

            # Extract title
            title = await self._extract_title(file_id)
            slug = slugify(title)[:200]  # Limit filename length
            LOGGER.info(f"{pdf_name} -> Title: {title}, Slug: {slug}")

            # Extract content sections
            title_and_abstract = await self._extract_title_and_abstract(file_id)
            LOGGER.info(f"{pdf_name} -> Title/Abstract: {repr(title_and_abstract[:50])}")

            last_section = await self._extract_last_section(file_id)
            LOGGER.info(f"{pdf_name} -> Last section: {repr(last_section[:50])}")

            # Create output directory and save artifacts
            output_dir = self._create_output_directory(slug)
            self._save_artifacts(output_dir, pdf_path, title)

            # Extract and save main content
            await self._extract_main_content(
                file_id,
                title_and_abstract,
                last_section,
                output_dir / "extracted.md",
            )

            LOGGER.info(f"Finished processing: {output_dir}")
            return slug, title

        except Exception as e:
            LOGGER.error(f"Error processing {pdf_path}: {e}", exc_info=True)
            return None

    def _get_pdf_files(self) -> List[Path]:
        """Get list of PDF files to process."""
        raw_dir = Path(self.raw_dir)
        if not raw_dir.exists():
            raise FileNotFoundError(f"Raw directory not found: {raw_dir}")

        pdf_files = list(raw_dir.glob("*.pdf"))
        LOGGER.info(f"Found {len(pdf_files)} PDF files to process")
        return pdf_files

    async def process_all_pdfs(self) -> None:
        """Process all PDF files in the raw directory."""
        pdf_files = self._get_pdf_files()

        if not pdf_files:
            LOGGER.info("No PDF files found to process")
            return

        tasks = [self.process_pdf(pdf_path) for pdf_path in pdf_files]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Count successful extractions
        successful = sum(1 for result in results if isinstance(result, tuple) and result[0] and result[1])
        LOGGER.info(f"Successfully extracted {successful} out of {len(pdf_files)} PDFs")


def validate_environment() -> None:
    """Validate required environment variables and settings."""
    if PROVIDER not in ["bailian"]:
        raise ValueError(f"PDF extraction requires bailian API, got: {PROVIDER}")

    raw_dir = os.getenv("RAW_DIR")
    if not raw_dir:
        raise ValueError("RAW_DIR environment variable not set")

    if not Path(raw_dir).exists():
        raise FileNotFoundError(f"Raw directory not found: {raw_dir}")


async def main() -> None:
    """Main entry point."""
    validate_environment()

    LOGGER.info(f"Starting PDF extraction: {os.getenv('RAW_DIR')} -> {VAULT_DIR}")

    extractor = PDFExtractor()
    await extractor.process_all_pdfs()

    LOGGER.info("All PDFs processed")


if __name__ == "__main__":
    asyncio.run(main())
