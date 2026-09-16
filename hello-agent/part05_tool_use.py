"""Part 5: Basic tool use with @function_tool.

You write a normal Python function. The decorator turns it into something the LLM can call:
    the function name  -> the tool's name
    the docstring      -> the tool's description (tells the LLM when to use it)
    the type hints     -> the JSON schema for the tool's parameters

Run:  uv run part05_tool_use.py
"""

from agents import Agent, Runner, function_tool

from config import llm_model


@function_tool
def get_weather(city: str) -> str:
    """Get the current weather for a given city."""
    return f"The weather in {city} is sunny and 25°C."


@function_tool
def add(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b


# One tool
weather_agent = Agent(
    name="Weather Assistant",
    instructions="You are a helpful assistant. Use tools when they help you answer.",
    model=llm_model,
    tools=[get_weather],
)
print("--- one tool ---")
print(Runner.run_sync(weather_agent, "What's the weather like in Karachi?").final_output)

# What happened:
#   1. run_sync sent the prompt plus the tool's schema to the LLM.
#   2. The LLM answered with "call get_weather with city='Karachi'" instead of text.
#   3. The SDK ran the Python function and sent the return value back as a new message.
#   4. The LLM read the tool result and wrote the final natural-language answer.
#   5. run_sync returned only once that whole loop finished.

# Try it: a second tool. The model calls both in the same turn and combines the results.
two_tool_agent = Agent(
    name="Assistant",
    instructions="Use tools when they help you answer.",
    model=llm_model,
    tools=[get_weather, add],
)
print("\n--- two tools ---")
print(Runner.run_sync(two_tool_agent, "What's 42 plus 58, and what's the weather in Lahore?").final_output)
