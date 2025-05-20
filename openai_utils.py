# openai_utils.py
from openai import OpenAI, APIConnectionError, RateLimitError, APIStatusError
import numpy as np
import time
from typing import Optional, List
from config import OPENAI_API_KEY, GPT_MODEL_SUMMARIZE, EMBEDDING_MODEL, GPT_MODEL_TAG, OPENAI_BASE_URL

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable not set.")

client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)

def get_embedding(text: str, model: str = EMBEDDING_MODEL) -> Optional[np.ndarray]:
    """Generates an embedding for the given text using OpenAI API."""
    if not text.strip(): # Avoid embedding empty strings
        return None
    try:
        text = text.replace("\n", " ") # API recommendation
        response = client.embeddings.create(input=[text], model=model)
        return np.array(response.data[0].embedding)
    except (APIConnectionError, RateLimitError, APIStatusError, Exception) as e:
        print(f"Error getting embedding: {e}")
        # Implement retry logic if necessary
        return None

def generate_completion(
    prompt: str,
    model: str = GPT_MODEL_SUMMARIZE,
    max_tokens: int = 1024,
    temperature: float = 0.5
) -> Optional[str]:
    """Generates a text completion using OpenAI Chat API."""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            n=1,
            stop=None
        )
        return response.choices[0].message.content.strip()
    except (APIConnectionError, RateLimitError, APIStatusError, Exception) as e:
        print(f"Error generating completion: {e}")
        # Implement retry logic if necessary
        return None
