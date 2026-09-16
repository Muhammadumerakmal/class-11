"""Part 11: Agents as tools: one agent calling another.

as_tool() wraps any agent so another agent can call it like get_weather. The orchestrator
stays in charge of the conversation; the specialist returns text and the orchestrator
writes the final answer around it.

Run:  uv run part11_agents_as_tools.py
"""

from agents import Agent, Runner, function_tool

from config import llm_model

spanish = Agent(
    name="Spanish Translator",
    instructions="Translate what the user says into Spanish. Only output Spanish.",
    model=llm_model,
)

summarizer = Agent(
    name="Summarizer",
    instructions="Summarize the given text in 2 short bullet points.",
    model=llm_model,
)

# Try it: full control with a function tool. When you need a turn limit, post-processing,
# or a different run config, write an ordinary @function_tool and call Runner.run inside it.
proofreader = Agent(
    name="Proofreader",
    instructions="Fix grammar and punctuation. Reply only with the corrected text.",
    model=llm_model,
)


@function_tool
async def proofread_text(text: str) -> str:
    """Fix grammar and punctuation; return only the corrected text."""
    result = await Runner.run(proofreader, text, max_turns=3)
    return str(result.final_output)


coach = Agent(
    name="Writing Coach",
    instructions=(
        "You help users improve messages.\n"
        "- If they ask for Spanish, call translate_to_spanish.\n"
        "- If they ask for a summary, call summarize_text.\n"
        "- If they ask to proofread, call proofread_text.\n"
        "- Otherwise, give a short tip yourself."
    ),
    model=llm_model,
    tools=[
        # The generated tool takes a single string parameter, input.
        # tool_description is what the orchestrator reads when choosing, like a docstring.
        spanish.as_tool(
            tool_name="translate_to_spanish",
            tool_description="Translate user text to Spanish.",
        ),
        summarizer.as_tool(
            tool_name="summarize_text",
            tool_description="Summarize text in 2 bullets.",
        ),
        proofread_text,
    ],
)

requests = [
    "Please translate to Spanish: I love hands-on examples.",
    "Summarize this: Agents are programs that use an LLM to decide which tools to call, "
    "run them, and loop until they have an answer. They can call other agents too.",
    "Proofread this: me and him goes to the store yesterday and buyed apple's",
]

for req in requests:
    result = Runner.run_sync(coach, req)
    # The reply comes from the coach, not from the specialist.
    print(f"\n> {req}\n{result.final_output}")
    tool_calls = [item.raw_item.name for item in result.new_items if item.type == "tool_call_item"]
    print(f"(tools fired: {tool_calls}, answered by: {result.last_agent.name})")

# Agent-as-tool is not a handoff. A handoff transfers the conversation; the specialist
# answers the user directly. Agent-as-tool borrows an answer and replies in its own voice.
