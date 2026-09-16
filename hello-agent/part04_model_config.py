"""Part 4: Model configuration at Agent, Run, and Global levels.

Three scopes for choosing which LLM backs an agent. Precedence, most specific wins:
    Agent level  >  Run level  >  Global level

The course always uses Agent-level configuration, so each agent in a multi-agent system
can use whichever model fits its job. Needs GEMINI_API_KEY to show the provider switch.

Run:  uv run part04_model_config.py
"""

import os
import sys

from agents import (
    Agent,
    AsyncOpenAI,
    OpenAIChatCompletionsModel,
    Runner,
    set_default_openai_api,
    set_default_openai_client,
    set_tracing_disabled,
)
from agents.run import RunConfig
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    sys.exit("GEMINI_API_KEY is empty in .env; this part needs a second provider to switch to.")

external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)
set_tracing_disabled(disabled=True)

# 1. Agent level: override per agent. Different agents in the same app can each have their own model=.
print("--- 1. Agent level ---")
agent_level = Agent(
    name="Assistant",
    instructions="You only respond in haikus.",
    model=OpenAIChatCompletionsModel(model="gemini-2.5-flash", openai_client=external_client),
)
print(Runner.run_sync(agent_level, "Hello, how are you.").final_output)

# 2. Run level: override for one call. The agent itself is unconfigured; the override applies
#    only to this specific Runner.run_sync(...) call. Useful for A/B testing a model.
print("\n--- 2. Run level ---")
config = RunConfig(
    model=OpenAIChatCompletionsModel(model="gemini-2.5-flash", openai_client=external_client),
    model_provider=external_client,
    tracing_disabled=True,
)
plain_agent = Agent(name="Assistant", instructions="You are a helpful assistant")
print(Runner.run_sync(plain_agent, "Hello, how are you.", run_config=config).final_output)

# 3. Global level: override for the whole process. Every agent created afterward uses this
#    client by default, unless it or its run overrides it locally.
print("\n--- 3. Global level ---")
set_default_openai_api("chat_completions")
set_default_openai_client(external_client)
global_agent = Agent(name="Assistant", instructions="You are a helpful assistant", model="gemini-2.5-flash")
print(Runner.run_sync(global_agent, "Hello, how are you.").final_output)
