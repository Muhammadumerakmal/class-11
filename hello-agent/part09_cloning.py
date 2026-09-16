"""Part 9: Cloning agents: one base, many variants.

agent.clone(...) copies an agent and overrides only the fields you name. Anything you
don't pass is carried over from the base.

Two traps:
    clone() is dataclasses.replace underneath, so it's a shallow copy. Lists are shared.
    A ModelSettings handed to clone() swaps the whole object; fields you leave out are dropped.

Run:  uv run part09_cloning.py
"""

from dataclasses import replace

from agents import Agent, ModelSettings, Runner, function_tool

from config import llm_model

base_agent = Agent(
    name="BaseAssistant",
    instructions="You are a helpful assistant.",
    model=llm_model,
    model_settings=ModelSettings(temperature=0.7),
)

creative_agent = base_agent.clone(
    name="CreativeAssistant",
    instructions="You are a creative writing assistant. Use vivid language.",
    model_settings=ModelSettings(temperature=0.9),
)

precise_agent = base_agent.clone(
    name="PreciseAssistant",
    instructions="You are a precise, factual assistant.",
    model_settings=ModelSettings(temperature=0.1),
)

for agent in (base_agent, creative_agent, precise_agent):
    print(f"\n{agent.name}:")
    print(Runner.run_sync(agent, "Describe a sunset.").final_output)


# The shallow-copy trap
@function_tool
def get_weather(city: str) -> str:
    """Get the weather for a city."""
    return f"The weather in {city} is sunny."


@function_tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


@function_tool
def calculate_area(length: float, width: float) -> float:
    """Area of a rectangle."""
    return length * width


print("\n--- shallow-copy trap ---")
base = Agent(name="Base", instructions="Be helpful.", tools=[get_weather])
shared_clone = base.clone(name="SharedClone")
base.tools.append(add)  # appended through the original...
print("shared_clone.tools:", len(shared_clone.tools), "(the clone grew a tool too)")
print("base.tools is shared_clone.tools:", base.tools is shared_clone.tools)

independent = base.clone(name="Independent", tools=[*base.tools])
base.tools.append(calculate_area)
print("independent.tools:", len(independent.tools), "(its own list, unaffected)")
print("base.tools is independent.tools:", base.tools is independent.tools)
# The rule: pass a fresh list whenever you want independence. Same for handoffs and guardrails.

# Try it: settings are replaced, not merged
print("\n--- settings are replaced, not merged ---")
base = Agent(
    name="Base",
    instructions="Be helpful.",
    model_settings=ModelSettings(temperature=0.7, max_tokens=500),
)
careless = base.clone(model_settings=ModelSettings(temperature=0.1))
print("careless max_tokens:", careless.model_settings.max_tokens, "(the 500 was dropped)")

careful = base.clone(model_settings=replace(base.model_settings, temperature=0.1))
print("careful max_tokens:", careful.model_settings.max_tokens, "(every other field kept)")
