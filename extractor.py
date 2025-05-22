import asyncio
from pathlib import Path
from paper_reader.config import MAX_CONCURRENT, PROVIDER, VAULT_DIR, LOGGER
from paper_reader.openai_utils import aclient
from paper_reader.utils import ensure_dir_exists, slugify
from paper_reader.prompts import load_prompt
import os

RAW_DIR = os.getenv("RAW_DIR")
SYS_PROMPT = load_prompt("prompts/pdf_extractor.md")
EXTRACTOR_TEMPERATURE = float(os.getenv("EXTRACTOR_TEMPREATURE", str(0.2)))
EXTRACTOR_MODEL = os.getenv("EXTRACTOR_MODEL", "qwen-long")

NAMING_PROMPT = """
What is the title of this paper? You should output the title only, no extra information or sentences.

**Examples:**

Attention Is All You Need

A Survey on Deep Learning for Image Super-Resolution

PhysGaussian: Physics-Integrated 3D Gaussians for Generative Dynamics
"""


async def process_pdf(pdf_path: Path, sem: asyncio.Semaphore):
    async with sem:
        try:
            pdf_path = Path(pdf_path)
            pdf_name = pdf_path.stem
            file_object = await aclient.files.create(
                file=pdf_path,
                purpose="file-extract",  # type: ignore
            )

            file_id = file_object.id
            LOGGER.info(f"{pdf_name} -> {file_id}")
            output = await aclient.chat.completions.create(
                model=EXTRACTOR_MODEL,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "system", "content": f"fileid://{file_id}"},
                    {
                        "role": "user",
                        "content": NAMING_PROMPT,
                    },
                ],
                temperature=0.2,
                stream=False,
                n=1,
            )

            assert output.choices[0].message.content is not None
            title = output.choices[0].message.content.strip()

            slug = slugify(title)[:200]  # avoid too long slug for filename.
            LOGGER.info(f"{pdf_name} -> Title: {title}, Slug: {slug}")
            output = await aclient.chat.completions.create(
                model=EXTRACTOR_MODEL,
                messages=[
                    {"role": "system", "content": SYS_PROMPT},
                    {"role": "system", "content": f"fileid://{file_id}"},
                    {
                        "role": "user",
                        "content": "Give me the text content of the pdf. (just the title and abstract)",
                    },
                ],
                stream=False,
                temperature=0.2,
            )

            assert output.choices[0].message.content is not None
            title_and_abstract = output.choices[0].message.content.strip()
            # print(f"token consumption: {output.usage.total_tokens}")
            LOGGER.info(f"{pdf_name} -> {repr(title_and_abstract[:50])}")
            if output.usage is not None:
                LOGGER.info(f"Consumption: {output.usage.total_tokens} tokens")

            # Get the last paragraph/section
            output = await aclient.chat.completions.create(
                model=EXTRACTOR_MODEL,
                messages=[
                    {"role": "system", "content": SYS_PROMPT},
                    {"role": "system", "content": f"fileid://{file_id}"},
                    {
                        "role": "user",
                        "content": "Give me the last paragraph or section of the pdf. (If appendix exists, it should be the appendix section, otherwise, its likely to be the conclusion section)",
                    },
                ],
                stream=False,
                temperature=0.2,
            )

            assert output.choices[0].message.content is not None
            last_paragraph_or_section = output.choices[0].message.content.strip()
            LOGGER.info(f"{pdf_name} -> {repr(last_paragraph_or_section[:50])}")
            if output.usage is not None:
                LOGGER.info(f"Consumption: {output.usage.total_tokens} tokens")

            main_pages_prompt = f"""
            You are going to extract the main pages.

            I have extracted the following information:
            ```markdown
            Title and Abstract:
            {title_and_abstract}

            Last Paragraph or Section:
            {last_paragraph_or_section}
            ```

            Please provide the main pages of the document:

            1. Your answer should begin with the title of the main pages.
            2. Your answer should end with the last paragraph or section I provided.
            3. You should begin with
            """
            output = await aclient.chat.completions.create(
                model=EXTRACTOR_MODEL,
                messages=[
                    {"role": "system", "content": SYS_PROMPT},
                    {"role": "system", "content": f"fileid://{file_id}"},
                    {"role": "user", "content": main_pages_prompt},
                ],
                temperature=EXTRACTOR_TEMPERATURE,
                stream=True,
                n=1,
            )
            output_dir = Path(f"{VAULT_DIR}/docs/{slug}")
            output_dir.mkdir(parents=True, exist_ok=True)
            with open(output_dir / "extracted.md", "w") as f:
                already_written = 0
                async for chunk in output:
                    last_already_written = already_written
                    if chunk.choices and chunk.choices[0].delta.content:
                        f.write(chunk.choices[0].delta.content)  # Write to file
                        already_written += len(chunk.choices[0].delta.content)
                    if last_already_written // 1000 < already_written // 1000:
                        LOGGER.info(f"Already writing {already_written} to {slug}.md")

            LOGGER.info(f"Finished writing {output_dir}")

        except Exception as e:
            LOGGER.error(f"Error processing {pdf_path}: {e}")
            return


async def main():
    assert RAW_DIR is not None
    dir_pdf_list = os.listdir(RAW_DIR)
    dir_pdf_list = [os.path.join(RAW_DIR, dir_pdf) for dir_pdf in dir_pdf_list if dir_pdf.endswith(".pdf")]

    LOGGER.info(f"Found {len(dir_pdf_list)} PDFs to process.")
    sem = asyncio.Semaphore(MAX_CONCURRENT if MAX_CONCURRENT > 0 else 8)

    tasks = []
    for pdf_path in dir_pdf_list:
        tasks.append(process_pdf(Path(pdf_path), sem))
    await asyncio.gather(*tasks)
    LOGGER.info("All PDFs processed.")


if __name__ == "__main__":
    assert PROVIDER in ["bailian"], "PDF extracting depends on bailian API."
    assert RAW_DIR is not None
    LOGGER.info(f"{RAW_DIR} -> {VAULT_DIR}")
    asyncio.run(main())
