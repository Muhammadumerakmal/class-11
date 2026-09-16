"""Part 1: OpenAI key integration (the default provider).

The Agents SDK is built by OpenAI, so OpenAI is the default provider.
No custom client wiring is needed, only OPENAI_API_KEY in .env.

Run:  uv run part01_openai_key.py
"""

from agents import Agent, Runner
from dotenv import find_dotenv, load_dotenv

# Loads .env so OPENAI_API_KEY is available as an environment variable
load_dotenv(find_dotenv())

# The SDK finds OPENAI_API_KEY automatically. No model= means the built-in default OpenAI model.
agent = Agent(
    name="Assistant",
    instructions="You are a helpful assistant.",
)

result = Runner.run_sync(agent, "Write a haiku about recursion in programming.")
print(result.final_output)

# Three things happened:
#   Agent(...)            defined who the agent is (a name + a system prompt via instructions)
#   Runner.run_sync(...)  sent the prompt to the LLM and waited for the result
#   result.final_output   the agent's final text response
