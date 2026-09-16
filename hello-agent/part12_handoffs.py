"""Part 12: Handoffs: transferring the conversation.

List the agents an agent may transfer to in handoffs=. The SDK turns each into a tool the
model can call (Study Coach becomes transfer_to_study_coach), so routing is tool selection.
After the handoff the specialist answers the user directly; the router's reply never
reaches the user.

Run:  uv run part12_handoffs.py
"""

import asyncio

from agents import Agent, Runner, handoff
from agents.items import HandoffCallItem, HandoffOutputItem

from config import llm_model

fitness_coach = Agent(
    name="Fitness Coach",
    instructions=(
        "You're a running coach. Ask 1-2 quick questions, then give a week plan. "
        "Keep it simple and encouraging. No medical advice."
    ),
    model=llm_model,
)

study_coach = Agent(
    name="Study Coach",
    instructions=(
        "You're a study planner. Ask for the current routine, then give a "
        "one-week schedule. Keep steps small and doable."
    ),
    model=llm_model,
)

router = Agent(
    name="Coach Router",
    instructions=(
        "Route the user:\n"
        "- running, workout, stamina -> hand off to the Fitness Coach.\n"
        "- exams, study plan, focus, notes -> hand off to the Study Coach.\n"
        "After the handoff the specialist continues the conversation."
    ),
    model=llm_model,
    # Both forms are legal: a bare agent, or handoff(agent) when you want to customise it.
    handoffs=[study_coach, handoff(fitness_coach)],
)


async def main():
    result = await Runner.run(router, "I want to run 5km in 8 weeks. Can you help?")
    print(result.final_output)

    # Proving the handoff happened: last_agent is the agent that actually answered.
    print("\nanswered by:", result.last_agent.name)
    for item in result.new_items:
        if isinstance(item, HandoffCallItem):
            print("HandoffCallItem   (the router asked):", item.raw_item.name)
        elif isinstance(item, HandoffOutputItem):
            print("HandoffOutputItem (the specialist accepted):",
                  item.source_agent.name, "->", item.target_agent.name)

    # Try it: keep talking to the specialist. The next turn goes straight to the coach,
    # not back through the router. to_input_list() hands you the conversation so far.
    follow_up = result.to_input_list() + [
        {"role": "user", "content": "Right now I jog about 2 km, 3 days a week."}
    ]
    second = await Runner.run(result.last_agent, follow_up)
    print("\n--- follow-up with", result.last_agent.name, "---")
    print(second.final_output)


if __name__ == "__main__":
    asyncio.run(main())

# The generated tool is named from the agent's name, lowercased with underscores.
# Rename an agent and you rename its handoff tool, which quietly invalidates any
# instructions that referred to the old name.
#
# handoff(agent, ...) also takes tool_name_override, tool_description_override,
# on_handoff (callback at the moment of transfer), input_type and input_filter.
