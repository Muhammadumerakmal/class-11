"""Part 2: Gemini key integration (a different provider).

The SDK only ships with OpenAI wiring, but it can talk to any provider that exposes an
OpenAI-compatible Chat Completions API. Gemini does. Needs GEMINI_API_KEY in .env.

Run:  uv run part02_gemini_key.py
"""

import os
import sys

from agents import Agent, AsyncOpenAI, OpenAIChatCompletionsModel, Runner, set_tracing_disabled
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    sys.exit("GEMINI_API_KEY is empty in .env. Get one at https://ai.google.dev/gemini-api/docs/api-key")

# OpenAI's tracing dashboard expects an OpenAI key; disable tracing for this Gemini-only script
set_tracing_disabled(disabled=True)

# 1. Point an OpenAI-compatible client at Gemini's endpoint instead of OpenAI's
external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

# 2. Wrap it as a Chat Completions model the SDK understands
llm_model = OpenAIChatCompletionsModel(
    model="gemini-2.5-flash",
    openai_client=external_client,
)

# 3. Hand that model to the Agent instead of using the default
agent = Agent(name="Assistant", model=llm_model)

result = Runner.run_sync(agent, "Welcome and motivate me to learn Agentic AI.")
print("AGENT RESPONSE:", result.final_output)

# Why this works:
#   AsyncOpenAI is just an HTTP client shaped for OpenAI's API contract. Gemini implements
#   that same contract at a different base_url, so the same client class works.
#   OpenAIChatCompletionsModel tells the Agent "call this client using the Chat Completions
#   format, with this specific model name."
#   Agent, Runner and .final_output are identical to Part 1. The provider is a pluggable model.
