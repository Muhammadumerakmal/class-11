"""Part 17: Run lifecycle hooks: watching every agent at once.

AgentHooks belongs to one agent and goes quiet at a handoff. RunHooks sits one level up,
is handed to the runner as hooks=, and watches every agent in the run.

The two classes don't line up:
    AgentHooks: on_start(ctx, agent) / on_end(ctx, agent, output) / on_handoff(ctx, agent, source)
    RunHooks:   on_agent_start(ctx, agent) / on_agent_end(ctx, agent, output) / on_handoff(ctx, from_agent, to_agent)
    on_tool_start, on_tool_end, on_llm_start, on_llm_end have the same shape on both.

Run:  uv run part17_run_hooks.py
"""

from agents import Agent, AgentHooks, RunHooks, Runner, function_tool

from config import llm_model


class SystemMonitor(RunHooks):
    def __init__(self):
        self.timeline = []

    async def on_agent_start(self, context, agent):
        self.timeline.append(f"start {agent.name}")

    async def on_tool_start(self, context, agent, tool):
        self.timeline.append(f"{agent.name} calls {tool.name}")

    async def on_handoff(self, context, from_agent, to_agent):
        self.timeline.append(f"handoff {from_agent.name} -> {to_agent.name}")

    async def on_agent_end(self, context, agent, output):
        self.timeline.append(f"end {agent.name}")


@function_tool
def get_weather(city: str) -> str:
    """A simple function to get the weather for a user."""
    return f"The weather for {city} is sunny."


news_agent = Agent(
    name="NewsAgent",
    instructions="Answer news questions.",
    model=llm_model,
)

weather_agent = Agent(
    name="WeatherAgent",
    instructions="Talk about weather. Let the NewsAgent handle news questions.",
    model=llm_model,
    tools=[get_weather],
    handoffs=[news_agent],
)

# One monitor for the whole run, passed to the runner, not to an agent.
monitor = SystemMonitor()
result = Runner.run_sync(weather_agent, "What's the latest news on AI coding tools?",
                         hooks=monitor)
print(result.final_output)
print("\nRunHooks timeline (spans both agents):")
for line in monitor.timeline:
    print(" ", line)


# Attach AgentHooks to the first agent instead and watch the trail stop at the handoff.
class AgentTrail(AgentHooks):
    def __init__(self):
        self.timeline = []

    async def on_start(self, context, agent):
        self.timeline.append(f"start {agent.name}")

    async def on_handoff(self, context, agent, source):
        # Fires on the hooks of the agent doing the handing off (source), with agent = destination.
        # This is the last event WeatherAgent's hooks see; NewsAgent's turn is not reported here.
        self.timeline.append(f"{source.name} handed off to {agent.name}")

    async def on_end(self, context, agent, output):
        self.timeline.append(f"end {agent.name}")


trail = AgentTrail()
weather_only = weather_agent.clone(hooks=trail)
result = Runner.run_sync(weather_only, "What's the latest news on AI coding tools?")
print("\nAgentHooks timeline on WeatherAgent only (stops at the handoff):")
for line in trail.timeline:
    print(" ", line)
print("answered by:", result.last_agent.name)

# Which one to reach for:
#   AgentHooks  one agent's own behaviour: a specialist you're tuning, a tool you suspect is slow
#   RunHooks    the request as a whole: total tokens and cost, an audit log that survives handoffs
# They compose. A run can carry a monitor while individual agents carry their own hooks.
