"""Part 3: Understanding Runner and asyncio.

Every LLM call is a network request, so the program sits waiting on I/O. asyncio lets it do
other work (or run several LLM calls at once) instead of blocking. The SDK is async natively.

Three ways to run an agent:
    Runner.run(...)           async, must be awaited. Real apps, servers, concurrent agents.
    Runner.run_sync(...)      plain function. Scripts, notebooks. Starts an event loop for you.
    Runner.run_streamed(...)  returns a streaming result for showing tokens as they arrive.

Run:  uv run part03_runner_asyncio.py
"""

import asyncio

from agents import Agent, Runner

from config import llm_model

# The async version, properly
haiku_agent = Agent(name="Assistant", instructions="You only respond in haikus.", model=llm_model)


async def single():
    # await pauses this coroutine until the LLM responds, without freezing the whole program
    result = await Runner.run(haiku_agent, "Tell me about recursion in programming.")
    print(result.final_output)


# Try it: run two agents concurrently
short_agent = Agent(name="Assistant", instructions="Answer in one short sentence.", model=llm_model)


async def concurrent():
    # Both prompts are in flight to the provider at the same time.
    # This is only possible because Runner.run is async.
    results = await asyncio.gather(
        Runner.run(short_agent, "What is Python?"),
        Runner.run(short_agent, "What is an AI agent?"),
    )
    for r in results:
        print(r.final_output)


async def main():
    print("--- single await ---")
    await single()
    print("\n--- two prompts with asyncio.gather ---")
    await concurrent()


if __name__ == "__main__":
    # asyncio.run creates the event loop, runs main() to completion, and shuts the loop down.
    # Runner.run_sync(agent, "...") from Parts 1 and 2 does exactly asyncio.run(Runner.run(...)).
    asyncio.run(main())
