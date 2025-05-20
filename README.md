# README - Research Article Summarization and Tag Management System

This project is a minimal Python-based system for processing research articles, generating summaries at various levels (article, section, TLDR), extracting relevant tags, and creating tag-specific descriptions and surveys. It uses OpenAI's API for text generation and embeddings, and stores content and vector data for future retrieval and iterative updates. The system is designed to be lightweight, using simple file-based storage instead of a complex database.

## Features

- **Article Processing**: Summarizes articles at different granularity levels (full summary, TLDR, per-section summaries).
- **Tag Extraction**: Identifies keywords (tags) from articles, representing problems, ideas, and technologies discussed.
- **Tag Management**: Generates Wikipedia-like descriptions and surveys of related papers for each tag.
- **Vector Embeddings**: Computes and stores embeddings for all generated content to enable similarity-based retrieval.
- **Iterative Updates**: Reuses existing summaries and embeddings on subsequent runs, allowing for iterative improvement.
- **Retrieval-Augmented Generation (RAG)**: Uses embeddings to fetch relevant context for generating tag descriptions and surveys.

## Directory Structure

```
.
├── main.py                      # Entry point to run the system
├── article_processor.py         # Logic for processing articles and generating summaries
├── tag_manager.py               # Logic for managing tags and generating descriptions/surveys
├── vector_store.py              # Functions for embedding storage and similarity search
├── openai_utils.py              # Helpers for interacting with OpenAI API
├── models.py                    # Data structures using TypedDict
├── config.py                    # Configuration and prompts
├── utils.py                     # Utility functions
└── vault/                       # Storage directory for processed data
    ├── docs/                    # Directory for article data
    │   └── paper_title_slug/    # Subdirectory per paper
    │       ├── extracted.md     # Original content (assumed to exist)
    │       ├── extracted.npz    # Embedding for original content
    │       ├── summarized.md    # Article-level summary
    │       ├── summarized.npz   # Embedding for summary
    │       ├── tldr.md          # TLDR summary
    │       ├── tldr.npz         # Embedding for TLDR
    │       ├── sections_summarized.md  # Concatenated section summaries
    │       └── sections_summarized.npz # Embedding for concatenated section summaries
    └── tags/                    # Directory for tag data
        └── tag_name_slug/       # Subdirectory per tag
            ├── description.md   # Wikipedia-like description
            ├── description.npz  # Embedding for description
            ├── survey.md        # Survey of related papers
            └── survey.npz       # Embedding for survey
```

## Prerequisites

- **Python 3.8+**
- **Dependencies**:
  - `openai`: For interacting with OpenAI's API for text generation and embeddings.
  - `numpy`: For handling vector embeddings.
- **OpenAI API Key**: You must have an API key to use OpenAI's services.

## Installation

1. **Clone or Set Up the Project Directory**:
   Ensure you have the project files in a directory of your choice.

2. **Install Required Packages**:
   ```bash
   pip install openai numpy
   ```

3. **Set OpenAI API Key**:
   Set your API key as an environment variable:
   ```bash
   export OPENAI_API_KEY="your_actual_api_key"
   ```
   Alternatively, on Windows, you can use:
   ```cmd
   set OPENAI_API_KEY=your_actual_api_key
   ```

## Usage

### Preparing Articles

1. **Create the `vault/docs/` Directory Structure**:
   - For each article, create a subdirectory in `vault/docs/` named after the article (preferably a slugified version of the title, e.g., `my-paper-title`).
   - Place the raw content of the article as `extracted.md` in its subdirectory.
   - Example:
     ```
     vault/
     └── docs/
         └── my-paper-title/
             └── extracted.md  # Your article content here
     ```

2. **Optional Section Separation**:
   - If your article content in `extracted.md` has sections, separate them with `<!-- SEPARATOR -->` to enable per-section summarization.

### Running the System

1. **Run the Main Script**:
   ```bash
   python main.py
   ```
   - The script will discover all subdirectories in `vault/docs/` containing an `extracted.md` file.
   - For each article, it generates summaries and embeddings, saving them in the respective article directory.
   - Tags are extracted, and for each tag, descriptions and surveys are generated and saved in `vault/tags/`.

2. **Iterative Runs**:
   - Subsequent runs of `python main.py` will reuse existing summaries and embeddings if they are present, avoiding redundant computation.
   - Tag descriptions and surveys may be updated if new articles are added or if forced regeneration is implemented.

### Dummy Data for Testing

- If no papers are found in `vault/docs/`, the system automatically creates a dummy paper (`example-paper-on-ai-ethics`) with sample content for demonstration purposes.

## Customization

- **Prompts**: Modify the prompts in `config.py` to adjust the style or focus of summaries, tag descriptions, or surveys.
- **Models**: Change the OpenAI model names in `config.py` if you prefer different models for summarization or embedding.
- **RAG**: Extend the Retrieval-Augmented Generation functionality in `vector_store.py` to improve context for summaries or tag content by adjusting `top_k` or similarity metrics.

## Limitations

- **Minimal Implementation**: This is a basic system without a full-fledged database. It relies on the file system for persistence, which may not scale well for large numbers of articles or tags.
- **Embedding Storage**: Embeddings are stored alongside markdown files as `.npz`, and similarity search is performed in-memory, which is not optimized for large datasets.
- **Error Handling**: Basic error handling for API calls; you may need to add retry logic for production use.
- **Section Summaries**: Individual section embeddings are computed, but storage could be more granular for better retrieval.

## Future Improvements

- Implement a lightweight database (e.g., SQLite) for managing metadata and relationships between articles and tags.
- Optimize vector storage and search using a proper vector database like FAISS or Annoy.
- Enhance RAG by fine-tuning similarity thresholds and context selection.
- Add a configuration file or command-line arguments for runtime options (e.g., force regeneration of summaries).

## License

This project is provided as-is for educational and personal use. Ensure you comply with OpenAI's usage policies when using their API.

## Contact

For questions or contributions, feel free to reach out or modify the codebase as needed for your use case.