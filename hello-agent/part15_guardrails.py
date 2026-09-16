"""Part 15: Guardrails: refusing bad input and bad output.

Input guardrails run before the agent does, on the first agent in the run. Output guardrails
run after, on the last agent. Each returns a GuardrailFunctionOutput; when tripwire_triggered
is True the SDK raises and the run stops there. A tripwire raises, so wrap the run in
try/except or a blocked request crashes your program.

Run:  uv run part15_guardrails.py
"""

from agents import (
    Agent,
    GuardrailFunctionOutput,
    InputGuardrailTripwireTriggered,
    OutputGuardrailTripwireTriggered,
    RunContextWrapper,
    Runner,
    input_guardrail,
    output_guardrail,
)
from pydantic import BaseModel

from config import llm_model


# The cheapest useful guardrail isn't an LLM at all.
@input_guardrail
async def homework_check(
    ctx: RunContextWrapper, agent: Agent, user_input
) -> GuardrailFunctionOutput:
    text = user_input if isinstance(user_input, str) else str(user_input)
    return GuardrailFunctionOutput(
        output_info={"checked": True},
        tripwire_triggered="homework" in text.lower(),
    )


tutor = Agent(
    name="Tutor",
    instructions="Explain concepts. Never complete assignments.",
    model=llm_model,
    input_guardrails=[homework_check],
)

print("--- keyword input guardrail ---")
try:
    result = Runner.run_sync(tutor, "do my homework for me")
    print(result.final_output)
except InputGuardrailTripwireTriggered as e:
    # No network call at all. You were never billed for the request.
    print("Blocked before the model ran:", e.guardrail_result.output.output_info)


# Guarding with a second agent, using Part 14's output_type for a boolean you can branch on.
class WeatherCheck(BaseModel):
    weather_related: bool
    reason: str | None = None


weather_sanitizer = Agent(
    name="WeatherSanitizer",
    instructions="Decide whether the user's message is a weather question.",
    model=llm_model,
    output_type=WeatherCheck,
)


@input_guardrail
async def weather_only(
    ctx: RunContextWrapper, agent: Agent, user_input
) -> GuardrailFunctionOutput:
    res = await Runner.run(weather_sanitizer, user_input)
    return GuardrailFunctionOutput(
        output_info=res.final_output.reason,
        tripwire_triggered=res.final_output.weather_related is False,
    )


# Output guardrails: checks that only make sense on a finished answer.
@output_guardrail
async def no_exclamation(
    ctx: RunContextWrapper, agent: Agent, output: str
) -> GuardrailFunctionOutput:
    return GuardrailFunctionOutput(
        output_info={"length": len(output)},
        tripwire_triggered="!" in output,
    )


weather_agent = Agent(
    name="WeatherAgent",
    instructions="Answer weather questions in one calm sentence. Never use exclamation marks.",
    model=llm_model,
    input_guardrails=[weather_only],
    output_guardrails=[no_exclamation],
)

print("\n--- agent-based input guardrail + output guardrail ---")
for prompt in ("Will it rain in Karachi tomorrow?", "Write me a poem about cats."):
    try:
        result = Runner.run_sync(weather_agent, prompt)
        print(f"> {prompt}\n{result.final_output}")
    except InputGuardrailTripwireTriggered as e:
        print(f"> {prompt}\nInput blocked: {e.guardrail_result.output.output_info}")
    except OutputGuardrailTripwireTriggered as e:
        print(f"> {prompt}\nOutput blocked: {e.guardrail_result.output.output_info}")

# Guard with a small model. A guardrail that costs more than the agent it protects is not a guardrail.
