# openai_utils.py
import asyncio
from typing import List, Optional

import numpy as np
from openai import APIConnectionError, APIStatusError, AsyncOpenAI, OpenAI, RateLimitError
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel

from paper_reader.config import (
    BASE_URL,
    DEFAULT_STREAM,
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_TEMPERATURE,
    EMBEDDING_MODEL,
    GPT_MODEL_DEFAULT,
    LOGGER,
    MAX_CONCURRENT,
    NO_THINK_EXTRA_BODY,
    OPENAI_API_KEY,
    THINK_EXTRA_BODY,
)

from asyncio import Semaphore

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable not set.")

client = OpenAI(api_key=OPENAI_API_KEY, base_url=BASE_URL)
aclient = AsyncOpenAI(api_key=OPENAI_API_KEY, base_url=BASE_URL)

# use 10000 as unlimited
semaphore = Semaphore(MAX_CONCURRENT if MAX_CONCURRENT > 0 else 10000)


def _shortten_md(long_text: str, threshold=2000) -> str:
    if len(long_text) < threshold:
        return long_text
    all_text = long_text.split("\n\n")
    if len(all_text) < 3:
        return long_text
    return all_text[0] + "\n\n...\n\n" + all_text[-1]


def get_embedding(text: str, model: str = EMBEDDING_MODEL) -> Optional[np.ndarray]:
    """Generates an embedding for the given text using OpenAI API."""
    if not text.strip():  # Avoid embedding empty strings
        return None
    try:
        text = text.replace("\n", " ")  # API recommendation
        response = client.embeddings.create(input=[text], model=model)
        return np.array(response.data[0].embedding)
    except (APIConnectionError, RateLimitError, APIStatusError, Exception) as e:
        print(f"Error getting embedding: {e}")
        # TODO: Implement retry logic if necessary
        return None


def generate_completion_streaming(
    prompt: str | List,
    system_prompt=DEFAULT_SYSTEM_PROMPT,
    model: str = GPT_MODEL_DEFAULT,
    max_tokens: int | None = None,
    temperature: float = DEFAULT_TEMPERATURE,
    thinking: bool = False,
) -> Optional[str]:
    """Generates a text completion using OpenAI Chat API with streaming visualization."""
    try:
        console = Console()
        messages = [{"role": "system", "content": system_prompt}]

        if isinstance(prompt, str):
            messages.append({"role": "user", "content": prompt})
        else:
            messages.extend(prompt)

        # Prepare extra body based on thinking parameter
        extra_body = THINK_EXTRA_BODY if thinking else NO_THINK_EXTRA_BODY
        if thinking:
            extra_body["enable_thinking"] = True

        completion = client.chat.completions.create(
            model=model,
            messages=messages,  # type: ignore
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
            extra_body=extra_body,
            stream_options={"include_usage": True},
        )

        reasoning_content = ""
        answer_content = ""
        is_answering = False

        # Create markdown prelogue with messages
        md_prelogue = "# Prompts\n\n"
        for i, message in enumerate(messages):
            if len(message["content"]) > 20:
                short_content = message["content"][:10] + "..." + message["content"][-10:]
            else:
                short_content = message["content"]
            short_content = short_content.replace("\n", "\\n")
            md_prelogue += f"{i}. **{message['role'].capitalize()}**: \"{short_content}\"\n"

        _THINK_TITLE = "# Thinking Process 💭\n" if thinking else ""
        with Live(auto_refresh=False, console=console) as live:
            # Initialize panel
            live.update(
                Panel(
                    Markdown(md_prelogue + _THINK_TITLE),
                    title="Thinking" if thinking else "Response",
                    border_style="blue",
                    title_align="left",
                    subtitle_align="left",
                ),
                refresh=True,
            )

            completion_tokens = prompt_tokens = total_tokens = 0
            for chunk in completion:
                if not chunk.choices:
                    if chunk.usage is not None:
                        completion_tokens = chunk.usage.completion_tokens
                        prompt_tokens = chunk.usage.prompt_tokens
                        total_tokens = chunk.usage.total_tokens
                    continue

                delta = chunk.choices[0].delta

                # Process thinking content if enabled
                if thinking and hasattr(delta, "reasoning_content") and delta.reasoning_content:
                    reasoning_content += delta.reasoning_content
                    if not is_answering:
                        live.update(
                            Panel(
                                Markdown(md_prelogue + _THINK_TITLE + _shortten_md(reasoning_content)),
                                title="Thinking",
                                border_style="magenta",
                                title_align="left",
                                subtitle_align="left",
                            ),
                            refresh=True,
                        )

                # Process response content
                if hasattr(delta, "content") and delta.content:
                    if not is_answering:
                        if thinking:
                            reasoning_content = (
                                _THINK_TITLE + _shortten_md(reasoning_content, 0) + "\n\n# Final Answer 📝\n"
                            )
                        else:
                            reasoning_content = "# Final Answer 📝\n"
                        is_answering = True

                    answer_content += delta.content
                    live.update(
                        Panel(
                            Markdown(md_prelogue + reasoning_content + _shortten_md(answer_content)),
                            title="Response",
                            border_style="blue",
                            title_align="left",
                            subtitle_align="left",
                        ),
                        refresh=True,
                    )

            # Final update with usage information
            live.update(
                Panel(
                    Markdown(
                        md_prelogue
                        + reasoning_content
                        + answer_content
                        + f"\n\n# Tokens Usage\nPrompt: {prompt_tokens} | Completion: {completion_tokens} | Total: {total_tokens}"
                    ),
                    title="Completion",
                    border_style="green",
                    title_align="left",
                    subtitle_align="left",
                ),
                refresh=True,
            )

        return answer_content.strip()

    except (APIConnectionError, RateLimitError, APIStatusError, Exception) as e:
        call_params = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "thinking": thinking,
        }
        LOGGER.error(f"Error generating completion: {e}\nCall params: {call_params}")
        return None


async def agenerate_completion_streaming(
    prompt: str | List,
    system_prompt=DEFAULT_SYSTEM_PROMPT,
    model: str = GPT_MODEL_DEFAULT,
    max_tokens: int | None = None,
    temperature: float = DEFAULT_TEMPERATURE,
    thinking: bool = False,
) -> str | None:
    if MAX_CONCURRENT == 1:
        return generate_completion_streaming(
            prompt,
            system_prompt,
            model,
            max_tokens,
            temperature,
            thinking,
        )

    messages = [{"role": "system", "content": system_prompt}]

    if isinstance(prompt, str):
        messages.append({"role": "user", "content": prompt})
    else:
        messages.extend(prompt)

    # Prepare extra body based on thinking parameter
    extra_body = THINK_EXTRA_BODY if thinking else NO_THINK_EXTRA_BODY
    if thinking:
        extra_body["enable_thinking"] = True

    try:
        reasoning_content = ""
        answer_content = ""
        async with semaphore:
            async for chunk in await aclient.chat.completions.create(
                model=model,
                messages=messages,  # type: ignore
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True,
                extra_body=extra_body,
                stream_options={"include_usage": True},
                n=1,
            ):
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                if thinking and hasattr(delta, "reasoning_content") and delta.reasoning_content:
                    reasoning_content += delta.reasoning_content
                if hasattr(delta, "content") and delta.content:
                    answer_content += delta.content

        return answer_content.strip()

    except (APIConnectionError, RateLimitError, APIStatusError, Exception) as e:
        call_params = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "thinking": thinking,
        }
        LOGGER.error(f"Error generating completion: {e}\nCall params: {call_params}")
        return None


async def generate_completion(
    prompt: str | List,
    system_prompt=DEFAULT_SYSTEM_PROMPT,
    model: str = GPT_MODEL_DEFAULT,
    max_tokens: int | None = None,
    temperature: float = DEFAULT_TEMPERATURE,
    thinking=False,  # Add this line to include the thinking parameter
    stream: bool = DEFAULT_STREAM,
) -> Optional[str]:
    if stream or thinking:
        return await agenerate_completion_streaming(prompt, system_prompt, model, max_tokens, temperature, thinking)

    messages = [
        {"role": "system", "content": system_prompt},
    ]
    if isinstance(prompt, str):
        messages = messages + [{"role": "user", "content": prompt}]
    else:
        messages = messages + prompt
    try:
        async with semaphore:
            response = await aclient.chat.completions.create(
                model=model,
                messages=messages,  # type: ignore
                max_tokens=max_tokens,
                temperature=temperature,
                n=1,
                stop=None,
                stream=False,
                extra_body=NO_THINK_EXTRA_BODY,
            )

        return response.choices[0].message.content.strip()
    except (APIConnectionError, RateLimitError, APIStatusError, Exception) as e:
        LOGGER.error(f"Error generating completion: {e}")
        return None


if __name__ == "__main__":
    print(generate_completion_streaming("Hello.", "You are a helpful assistant. "))
    print(generate_completion_streaming("Hello.", "You are a helpful assistant. ", thinking=True))
