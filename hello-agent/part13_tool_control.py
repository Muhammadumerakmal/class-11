"""Part 13: Advanced tool control: stopping, gating, and failing well.

    tool_use_behavior  when the loop stops ("run_llm_again" default, "stop_on_first_tool", StopAtTools)
    max_turns          caps LLM calls; the SDK raises MaxTurnsExceeded instead of looping forever
    is_enabled         whether a tool is offered to the model at all; can read the run context
    returning errors   a tool that raises kills the run; a tool that returns its error keeps it alive

Run:  uv run part13_tool_control.py
"""

import asyncio
from dataclasses import dataclass

from agents import Agent, RunContextWrapper, Runner, StopAtTools, function_tool
from agents.exceptions import MaxTurnsExceeded

from config import llm_model


# Stopping the loop early
@function_tool
def get_weather(city: str) -> str:
    """A simple function to get the weather for a user."""
    return "Sunny"


@function_tool
def get_travel_plan(city: str) -> str:
    """Plan travel for the given city."""
    return "Travel plan is not available"


travel_agent = Agent(
    name="TravelAgent",
    instructions="You are a helpful assistant.",
    model=llm_model,
    tools=[get_weather, get_travel_plan],
    tool_use_behavior=StopAtTools(stop_at_tool_names=["get_travel_plan"]),
)

print("--- StopAtTools ---")
result = Runner.run_sync(travel_agent, "Make me a travel plan for Lahore")
# final_output is now the tool's own string, word for word. The model never saw it.
print(repr(result.final_output))

# The safety net: max_turns. A tool call plus the model reading the result is two turns,
# so max_turns=1 stops before the agent can ever use a tool.
print("\n--- max_turns ---")
plain_agent = Agent(
    name="Researcher",
    instructions="Use tools when they help.",
    model=llm_model,
    tools=[get_weather, get_travel_plan],
)
try:
    result = Runner.run_sync(plain_agent, "What's the weather in Lahore?", max_turns=1)
    print(result.final_output)
except MaxTurnsExceeded:
    print("Hit the turn limit. Stopping instead of looping forever.")


# Tools that appear and disappear: is_enabled builds permissions from the Part 7 context.
@dataclass
class UserContext:
    user_id: str
    subscription_tier: str = "free"


def premium_only(ctx: RunContextWrapper[UserContext], agent: Agent) -> bool:
    return ctx.context.subscription_tier in ("premium", "enterprise")


@function_tool(is_enabled=premium_only)
def deep_research(topic: str) -> str:
    """Run an expensive deep research pass on a topic."""
    return f"Deep research on {topic}: agents, tools, handoffs, guardrails."


# Failing where the model can see it
@function_tool
def divide(a: int, b: int) -> str:
    """Divide two numbers."""
    try:
        return str(a / b)
    except ZeroDivisionError:
        return "Error: cannot divide by zero. Ask the user for a different number."


gated_agent = Agent[UserContext](
    name="GatedAgent",
    instructions="Use deep_research if you have it, otherwise say you can't research deeply.",
    model=llm_model,
    tools=[deep_research, divide],
)


async def show_tools_and_run(tier: str):
    ctx = UserContext(user_id="u1", subscription_tier=tier)
    wrapper = RunContextWrapper(context=ctx)
    offered = [t.name for t in await gated_agent.get_all_tools(wrapper)]
    print(f"\n[{tier}] tools offered to the model: {offered}")
    result = await Runner.run(gated_agent, "Do deep research on AI agents.", context=ctx)
    print(f"[{tier}] {result.final_output}")


async def main():
    print("\n--- is_enabled ---")
    await show_tools_and_run("free")
    await show_tools_and_run("premium")
    # A disabled tool isn't refused. It's never mentioned, so the model can't ask for it.

    print("\n--- a tool that returns its error ---")
    result = await Runner.run(gated_agent, "What is 10 divided by 0?",
                              context=UserContext(user_id="u1"))
    print(result.final_output)


asyncio.run(main())
