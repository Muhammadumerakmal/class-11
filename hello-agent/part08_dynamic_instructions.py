"""Part 8: Dynamic instructions: a system prompt that changes.

instructions= also accepts a function. It runs at the start of each turn and returns the
string to use as that turn's system prompt. The function must take exactly two parameters,
(context, agent), or the SDK raises TypeError.

Run:  uv run part08_dynamic_instructions.py
"""

from dataclasses import dataclass

from agents import Agent, RunContextWrapper, Runner

from config import llm_model


@dataclass
class UserInfo:
    name: str
    uid: int


# ctx.context is the object passed as context= in Part 7. agent is the agent being run.
def special_prompt(ctx: RunContextWrapper[UserInfo], agent: Agent) -> str:
    return (
        f"You are a math expert. User: {ctx.context.name}, Agent: {agent.name}. "
        "Please assist with math-related queries."
    )


math_agent = Agent[UserInfo](
    name="Genius",
    instructions=special_prompt,
    model=llm_model,
)

user = UserInfo(name="Ali", uid=123)
print("--- function instructions ---")
print(Runner.run_sync(math_agent, "What is 17 × 23?", context=user).final_output)


# Try it: instructions that remember. Any callable object works, which is how you keep state.
class StatefulInstructions:
    def __init__(self):
        self.count = 0

    def __call__(self, ctx: RunContextWrapper, agent: Agent) -> str:
        self.count += 1
        if self.count == 1:
            return "You are a learning assistant. First interaction. Be welcoming."
        return f"You are a learning assistant. Interaction #{self.count}. Be brief."


stateful_agent = Agent(name="Stateful", instructions=StatefulInstructions(), model=llm_model)
print("\n--- callable class instructions ---")
for _ in range(3):
    print(Runner.run_sync(stateful_agent, "Tell me about AI in one line").final_output)

# RunContextWrapper carries .context, .usage and .turn_input. It does not carry past messages.
# If you want to react to conversation length, count it yourself like StatefulInstructions does.
