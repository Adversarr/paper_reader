# Paper Reader

A minimal LLM based paper reader.

> **Note**:
> 
> 1. This project is still in its early stages and may require some adjustments to work seamlessly with different papers and LLMs.
> 2. The current implementation relies on Aliyun Bailian API to extract contents from PDF files. (`extractor.py`)

## Usage

### Setup

Install `uv`. Then setup the venv:

```bash
uv venv && uv sync && uv pip install -e .
```

Setup your API keys in `.env` file:

```sh
EMBEDDING_API_KEY=sk-XXXXXXXXXXXXXXXXXXXXXXXXXXX
API_KEY=sk-XXXXXXXXXXXXXXXXXXXXXXXXXXX
EMBEDDING_PROVIDER=bailian # Only bailian is supported for now
PRIVIDER=bailian # bailian, openai, openrouter, volcengine, custom
# BASE_URL=https://url.to.your.provider/v1 # Only needed if using a custom provider
```

### Running the Paper Reader

#### Extraction Stage: Extracting Contents from PDFs

Prepare all the papers in raw directory. For example:
```sh
❯ tree raw
raw
├── diffpd.pdf
├── flow_mixer.pdf
├── mash.pdf
├── nerf.pdf
└── physgaussian.pdf

1 directory, 5 files
```

To run the paper reader, use the following command:

```sh
$ RAW_DIR=raw VAULT_DIR=vault python extractor.py
```

If you triggers the maximum TPM/QPM limit, you can set the envvar:

```sh
$ MAX_CONCURRENT=<SMALL_INTEGER> RAW_DIR=raw VAULT_DIR=vault python extractor.py
```

You will see the outputs in the `vault` directory.

#### Reading Stage: Reading and Summarizing Papers

To read and summarize the papers, use the following command:

```sh
$ MAX_CONCURRENT=4 DEFAULT_REBUILD=true DEFAULT_MODEL=qwen-plus GPT_MODEL_FAST=qwen-turbo DEFAULT_THINKING=true python main.py
```
