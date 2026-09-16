"""Part 6: Model settings: temperature, tool choice, and length.

Part 4 decided which model runs the agent. ModelSettings decides how that model behaves.
    temperature  the creativity dial. Low is focused and repeatable, high is varied.
    tool_choice  whether the model may ("auto"), must ("required"), or must not ("none") call a tool.
    max_tokens   a hard ceiling on answer length. It cuts off, it does not ask for brevity.

Run:  uv run part06_model_settings.py
"""

from agents import Agent, ModelSettings, Runner, function_tool

from config import PROVIDER, llm_model

# Temperature. Gemini accepts up to 2.0; OpenAI models accept up to 2.0 as well.
question = "Tell me about AI in 2 sentences."
print("=== temperature ===")
for temperature in (0.1, 1.9):
    agent = Agent(
        name="Assistant",
        instructions="You are a helpful assistant.",
        model=llm_model,
        model_settings=ModelSettings(temperature=temperature),
    )
    print(f"\ntemperature={temperature}:")
    print(Runner.run_sync(agent, question).final_output)


# Forcing or forbidding tool use. tool_choice overrides the model's own judgement.
@function_tool
def calculate_area(length: float, width: float) -> str:
    """Calculate the area of a rectangle."""
    return f"Area = {length} × {width} = {length * width} square units"


question = "What's the area of a 5x3 rectangle?"
print("\n=== tool_choice ===")
for choice in ("auto", "required", "none"):
    agent = Agent(
        name=f"{choice} agent",
        instructions="You are a helpful assistant.",
        model=llm_model,
        tools=[calculate_area],
        model_settings=ModelSettings(tool_choice=choice),
    )
    print(f"\ntool_choice={choice}:")
    print(Runner.run_sync(agent, question).final_output)

# "auto"      the model decides (the default, what Part 5 was doing)
# "required"  the model must call a tool before it may answer
# "none"      the model sees the schema but is forbidden from calling it; it does the maths itself

# Try it: cap the answer length. If the answer needs more room, it stops mid-sentence.
print("\n=== max_tokens ===")
brief_agent = Agent(
    name="Brief Assistant",
    instructions="You are a helpful assistant.",
    model=llm_model,
    model_settings=ModelSettings(temperature=0.2, max_tokens=100),
)
print(Runner.run_sync(brief_agent, "Explain how neural networks learn.").final_output)
print(f"\n(provider: {PROVIDER})")
