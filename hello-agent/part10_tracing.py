"""Part 10: Tracing: seeing what your agent actually did.

A trace is one complete workflow. A span is one step inside it (an LLM call, a tool call,
one agent's turn). Tracing is on by default. The dashboard is OpenAI's, so exporting traces
needs an OpenAI key even when inference runs on Gemini. config.py already calls
set_tracing_export_api_key with OPENAI_API_KEY.

After running, open https://platform.openai.com/traces

Run:  uv run part10_tracing.py
"""

import asyncio

from agents import Agent, Runner, function_tool, trace

from config import OPENAI_API_KEY, llm_model

if not OPENAI_API_KEY:
    print("OPENAI_API_KEY is not set; the SDK will skip trace export and carry on.")


@function_tool
def get_weather(city: str) -> str:
    """A simple function to get the weather for a user."""
    return f"The weather for {city} is sunny."


agent = Agent(
    name="WeatherAgent",
    instructions="You are a helpful assistant.",
    model=llm_model,
    tools=[get_weather],
)

# One run, one trace: the run, the tool call, and the model's second pass over the result.
print("--- single run ---")
result = Runner.run_sync(agent, "What's the weather in Karachi?")
print(result.final_output)


# Try it: group several runs into one trace. By default every Runner.run(...) is its own trace.
async def main():
    with trace("Joke workflow"):
        first = await Runner.run(agent, "Tell me a joke")
        second = await Runner.run(agent, f"Rate this joke: {first.final_output}")
        print(f"Joke: {first.final_output}")
        print(f"Rating: {second.final_output}")


print("\n--- two runs under one trace ---")
asyncio.run(main())

# Turning it back off:
#   set_tracing_disabled(True)                         whole process
#   RunConfig(tracing_disabled=True)                   one run
#   RunConfig(trace_include_sensitive_data=False)      timeline only, no prompts or outputs
