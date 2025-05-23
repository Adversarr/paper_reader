# openai_utils.py
import asyncio
from asyncio import Semaphore
from typing import List, Optional

import numpy as np
from openai import (
    APIConnectionError,
    APIStatusError,
    AsyncOpenAI,
    OpenAI,
    RateLimitError,
)
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from paper_reader.config import (
    BASE_URL,
    DEFAULT_STREAM,
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_TEMPERATURE,
    EMBEDDING_API_KEY,
    EMBEDDING_BASE_URL,
    EMBEDDING_MODEL,
    MODEL_DEFAULT,
    LOGGER,
    MAX_CONCURRENT,
    NO_THINK_EXTRA_BODY,
    OPENAI_API_KEY,
    THINK_EXTRA_BODY,
)

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable not set.")

client = OpenAI(api_key=OPENAI_API_KEY, base_url=BASE_URL)
client_embedding = OpenAI(api_key=EMBEDDING_API_KEY, base_url=EMBEDDING_BASE_URL)
aclient = AsyncOpenAI(api_key=OPENAI_API_KEY, base_url=BASE_URL)

# use 10000 as unlimited
semaphore = Semaphore(MAX_CONCURRENT if MAX_CONCURRENT > 0 else 10000)

TOTAL_TOKENS_PROMPT: dict[str, int] = {}
TOTAL_TOKENS_COMPLETION: dict[str, int] = {}
TOTAL_TOKENS_TOTAL: dict[str, int] = {}
TOTAL_CALL_COUNTER: dict[str, int] = {}


async def rich_print_consumption_table():
    async with semaphore:
        console = Console()
        table = Table(title="API Usage Statistics")

        table.add_column("Model", style="cyan")
        table.add_column("API Calls", style="magenta")
        table.add_column("Prompt Tokens", style="green")
        table.add_column("Completion Tokens", style="blue")
        table.add_column("Total Tokens", style="red")

        total_calls = 0
        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_total_tokens = 0

        # Sort models alphabetically for consistent display
        models = sorted(
            set(TOTAL_CALL_COUNTER.keys())
            | set(TOTAL_TOKENS_PROMPT.keys())
            | set(TOTAL_TOKENS_COMPLETION.keys())
            | set(TOTAL_TOKENS_TOTAL.keys())
        )

        for model in models:
            calls = TOTAL_CALL_COUNTER.get(model, 0)
            prompt_tokens = TOTAL_TOKENS_PROMPT.get(model, 0)
            completion_tokens = TOTAL_TOKENS_COMPLETION.get(model, 0)
            total_tokens = TOTAL_TOKENS_TOTAL.get(model, 0)

            total_calls += calls
            total_prompt_tokens += prompt_tokens
            total_completion_tokens += completion_tokens
            total_total_tokens += total_tokens

            table.add_row(
                model,
                str(calls),
                str(prompt_tokens),
                str(completion_tokens),
                str(total_tokens),
            )

        # Add a summary row
        table.add_row(
            "TOTAL",
            str(total_calls),
            str(total_prompt_tokens),
            str(total_completion_tokens),
            str(total_total_tokens),
            style="bold",
        )

        console.print(table)


def _shortten_md(long_text: str, threshold=2000) -> str:
    if len(long_text) < threshold:
        return long_text

    all_text = long_text.split("\n\n")
    if len(all_text) < 3:
        return (
            long_text[: threshold // 2] + "\n\n...\n\n" + long_text[-threshold // 2 :]
        )

    return all_text[0] + "\n\n...\n\n" + all_text[-1]


def get_embedding(text: str, model: str = EMBEDDING_MODEL) -> Optional[np.ndarray]:
    """Generates an embedding for the given text using OpenAI API."""
    if not text.strip():
        LOGGER.warning("Cannot generate embedding for empty text")
        return None
    try:
        # Track usage
        if model not in TOTAL_CALL_COUNTER:
            TOTAL_CALL_COUNTER[model] = 0
        TOTAL_CALL_COUNTER[model] += 1

        response = client_embedding.embeddings.create(
            model=model,
            input=[text],
        )

        # Update token usage statistics
        if hasattr(response, "usage") and response.usage is not None:
            prompt_tokens = response.usage.prompt_tokens
            if model not in TOTAL_TOKENS_PROMPT:
                TOTAL_TOKENS_PROMPT[model] = 0
                TOTAL_TOKENS_TOTAL[model] = 0
            TOTAL_TOKENS_PROMPT[model] += prompt_tokens
            TOTAL_TOKENS_TOTAL[model] += prompt_tokens

        return np.array(response.data[0].embedding)

    except (APIConnectionError, RateLimitError, APIStatusError) as e:
        LOGGER.error(f"API error while generating embedding: {e}")
        return None
    except Exception as e:
        LOGGER.error(f"Unexpected error while generating embedding: {e}")
        return None


def generate_completion_streaming(
    prompt: str | List,
    system_prompt=DEFAULT_SYSTEM_PROMPT,
    model: str = MODEL_DEFAULT,
    max_tokens: int | None = None,
    temperature: float = DEFAULT_TEMPERATURE,
    thinking: bool = False,
) -> Optional[str]:
    """Generates a text completion using OpenAI Chat API with streaming visualization."""
    try:
        messages = []
        messages.append({"role": "system", "content": system_prompt})

        if isinstance(prompt, str):
            messages.append({"role": "user", "content": prompt})
        else:
            messages.extend(prompt)

        console = Console()
        panel_content = ""

        # Prepare extra body based on thinking parameter
        extra_body = THINK_EXTRA_BODY if thinking else NO_THINK_EXTRA_BODY

        # Track API usage
        if model not in TOTAL_CALL_COUNTER:
            TOTAL_CALL_COUNTER[model] = 0
        TOTAL_CALL_COUNTER[model] += 1

        # Display thinking process if enabled
        if thinking:
            console.print(Panel(Markdown("Thinking..."), title="Thinking Process"))

        # Create the stream
        stream = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
            stream_options={"include_usage": True},
            extra_body=extra_body,
        )

        prompt_tokens = completion_tokens = total_tokens = 0
        with Live(
            Panel(Markdown(""), title="Generating Response"), refresh_per_second=10
        ) as live:
            for chunk in stream:
                if chunk.usage:
                    prompt_tokens = chunk.usage.prompt_tokens
                    completion_tokens = chunk.usage.completion_tokens
                    total_tokens = chunk.usage.total_tokens

                if chunk.choices and chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    panel_content += content
                    live.update(
                        Panel(Markdown(panel_content), title="Generating Response")
                    )

        if model not in TOTAL_TOKENS_PROMPT:
            TOTAL_TOKENS_PROMPT[model] = 0
            TOTAL_TOKENS_COMPLETION[model] = 0
            TOTAL_TOKENS_TOTAL[model] = 0

        TOTAL_TOKENS_PROMPT[model] += prompt_tokens
        TOTAL_TOKENS_COMPLETION[model] += completion_tokens
        TOTAL_TOKENS_TOTAL[model] += total_tokens

        return panel_content

    except (APIConnectionError, RateLimitError, APIStatusError) as e:
        LOGGER.error(f"API error while generating completion: {e}")
        return None
    except Exception as e:
        LOGGER.error(f"Unexpected error while generating completion: {e}")
        return None


async def agenerate_completion_streaming(
    prompt: str | List,
    system_prompt=DEFAULT_SYSTEM_PROMPT,
    model: str = MODEL_DEFAULT,
    max_tokens: int | None = None,
    temperature: float = DEFAULT_TEMPERATURE,
    thinking: bool = False,
) -> str | None:
    if MAX_CONCURRENT == 1:
        return generate_completion_streaming(
            prompt, system_prompt, model, max_tokens, temperature, thinking
        )

    async with semaphore:
        messages = []
        messages.append({"role": "system", "content": system_prompt})

        if isinstance(prompt, str):
            messages.append({"role": "user", "content": prompt})
        else:
            messages.extend(prompt)

        # Prepare extra body based on thinking parameter
        extra_body = THINK_EXTRA_BODY if thinking else NO_THINK_EXTRA_BODY
        console = Console()

        if thinking:
            console.print(Panel(Markdown("Thinking..."), title="Thinking Process"))

        try:
            # Track API usage
            if model not in TOTAL_CALL_COUNTER:
                TOTAL_CALL_COUNTER[model] = 0
            TOTAL_CALL_COUNTER[model] += 1

            panel_content = ""

            # Create the stream
            stream = await aclient.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True,
                extra_body=extra_body,
                stream_options={"include_usage": True},
            )

            completion_tokens = prompt_tokens = total_tokens = -1

            async for chunk in stream:
                if chunk.usage:
                    prompt_tokens = chunk.usage.prompt_tokens
                    completion_tokens = chunk.usage.completion_tokens
                    total_tokens = chunk.usage.total_tokens

                if chunk.choices and chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    panel_content += content

            if model not in TOTAL_TOKENS_PROMPT:
                TOTAL_TOKENS_PROMPT[model] = 0
                TOTAL_TOKENS_COMPLETION[model] = 0
                TOTAL_TOKENS_TOTAL[model] = 0

            TOTAL_TOKENS_PROMPT[model] += prompt_tokens
            TOTAL_TOKENS_COMPLETION[model] += completion_tokens
            TOTAL_TOKENS_TOTAL[model] += total_tokens

            return panel_content

        except (APIConnectionError, RateLimitError, APIStatusError) as e:
            LOGGER.error(f"API error while generating completion: {e}")
            return None
        except Exception as e:
            LOGGER.error(f"Unexpected error while generating completion: {e}")
            return None


async def generate_completion(
    prompt: str | List,
    system_prompt=DEFAULT_SYSTEM_PROMPT,
    model: str = MODEL_DEFAULT,
    max_tokens: int | None = None,
    temperature: float = DEFAULT_TEMPERATURE,
    thinking=False,
    stream: bool = DEFAULT_STREAM,
) -> Optional[str]:
    if stream or thinking:
        return await agenerate_completion_streaming(
            prompt, system_prompt, model, max_tokens, temperature, thinking
        )

    async with semaphore:
        messages = []
        messages.append({"role": "system", "content": system_prompt})

        if isinstance(prompt, str):
            messages.append({"role": "user", "content": prompt})
        else:
            messages.extend(prompt)

        try:
            # Track API usage
            if model not in TOTAL_CALL_COUNTER:
                TOTAL_CALL_COUNTER[model] = 0
            TOTAL_CALL_COUNTER[model] += 1

            response = await aclient.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=False,
            )

            # Update token usage statistics
            if hasattr(response, "usage") and response.usage is not None:
                if model not in TOTAL_TOKENS_PROMPT:
                    TOTAL_TOKENS_PROMPT[model] = 0
                    TOTAL_TOKENS_COMPLETION[model] = 0
                    TOTAL_TOKENS_TOTAL[model] = 0

                TOTAL_TOKENS_PROMPT[model] += response.usage.prompt_tokens
                TOTAL_TOKENS_COMPLETION[model] += response.usage.completion_tokens
                TOTAL_TOKENS_TOTAL[model] += response.usage.total_tokens

            # Extract and return content
            if response.choices and response.choices[0].message.content:
                return response.choices[0].message.content
            return None

        except (APIConnectionError, RateLimitError, APIStatusError) as e:
            LOGGER.error(f"API error while generating completion: {e}")
            return None
        except Exception as e:
            LOGGER.error(f"Unexpected error while generating completion: {e}")
            return None


# if __name__ == "__main__":
#     print(generate_completion_streaming("Hello.", "You are a helpful assistant. "))
#     print(generate_completion_streaming("Hello.", "You are a helpful assistant. ", thinking=True))
