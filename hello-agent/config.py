"""Shared model setup used by Parts 3 to 17.

The guide builds `llm_model` once in Part 2 and reuses it everywhere after.
This module does the same so every part file can just `from config import llm_model`.

- If GEMINI_API_KEY is set, llm_model is Gemini via the OpenAI-compatible endpoint (Part 2).
- If not, llm_model is None, which makes Agent(model=None) use the SDK's default OpenAI model (Part 1).

Tracing (Part 10) uploads to OpenAI's dashboard and needs an OpenAI key even when
inference runs on Gemini. If the key is present it's registered for trace export;
if not, tracing is disabled so the SDK stops warning about it.
"""

import os

from agents import (
    AsyncOpenAI,
    OpenAIChatCompletionsModel,
    set_tracing_disabled,
    set_tracing_export_api_key,
)
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

if GEMINI_API_KEY:
    # 1. Point an OpenAI-compatible client at Gemini's endpoint instead of OpenAI's
    external_client = AsyncOpenAI(
        api_key=GEMINI_API_KEY,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )
    # 2. Wrap it as a Chat Completions model the SDK understands
    llm_model = OpenAIChatCompletionsModel(
        model="gemini-2.5-flash",
        openai_client=external_client,
    )
    PROVIDER = "gemini"
else:
    external_client = None
    llm_model = None
    PROVIDER = "openai (default)"

if OPENAI_API_KEY:
    # Inference may run on Gemini; this key is only used to upload traces
    set_tracing_export_api_key(OPENAI_API_KEY)
else:
    set_tracing_disabled(True)
