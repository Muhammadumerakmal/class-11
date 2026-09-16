"""Part 16: Lifecycle hooks: watching the loop from inside.

AgentHooks has one method per stage. Override the ones you care about and hand an instance
to the agent. The full set: on_start, on_llm_start, on_llm_end, on_tool_start, on_tool_end,
on_handoff, on_end.

on_start fires once, when this agent becomes responsible for answering.
on_llm_start fires every time the agent calls the model (twice when a tool is used).
Keep hooks fast: they run inside the loop.

Run:  uv run part16_agent_hooks.py
"""

from agents import Agent, AgentHooks, Runner, function_tool

from config import llm_model


class LoudHooks(AgentHooks):
    def __init__(self, label: str):
        self.label = label

    async def on_start(self, context, agent):
        print(f"[{self.label}] {agent.name} is now responsible for the answer")

    async def on_llm_start(self, context, agent, system_prompt, input_items):
        print(f"[{self.label}] calling the model ({len(input_items)} input items)")

    async def on_llm_end(self, context, agent, response):
        print(f"[{self.label}] model responded")

    async def on_tool_start(self, context, agent, tool):
        print(f"[{self.label}] calling {tool.name}")

    async def on_tool_end(self, context, agent, tool, result):
        print(f"[{self.label}] {tool.name} returned {result}")

    async def on_end(self, context, agent, output):
        print(f"[{self.label}] finished")


@function_tool
def get_weather(city: str) -> str:
    """A simple function to get the weather for a user."""
    return f"The weather for {city} is sunny."


agent = Agent(
    name="WeatherAgent",
    instructions="You are a helpful assistant.",
    model=llm_model,
    tools=[get_weather],
    hooks=LoudHooks("weather"),
)

result = Runner.run_sync(agent, "What's the weather in Karachi?")
print("\nfinal:", result.final_output)

# The order you see is the two-pass loop from Part 5:
#   on_start -> on_llm_start -> on_llm_end -> on_tool_start -> on_tool_end
#   -> on_llm_start -> on_llm_end -> on_end
#
# AgentHooks belongs to one agent. RunHooks (Part 17) covers the whole run, handoffs included.
# The two classes don't share method names: AgentHooks has on_start / on_end,
# RunHooks has on_agent_start / on_agent_end. Override the wrong pair and nothing fires.
