"""Part 7: Local context: data your tools see and the LLM doesn't.

You define a plain Python object, hand it to Runner.run(..., context=...), and the SDK wraps
it in a RunContextWrapper that every tool in that run receives as its first parameter.
The context is never sent to the LLM.

Run:  uv run part07_local_context.py
"""

import asyncio
from dataclasses import dataclass

from agents import Agent, RunContextWrapper, Runner, function_tool

from config import llm_model


@dataclass
class UserInfo:
    name: str
    uid: int
    location: str = "Pakistan"


@function_tool
async def fetch_user_age(wrapper: RunContextWrapper[UserInfo]) -> str:
    """Returns the age of the user."""
    return f"User {wrapper.context.name} is 47 years old"


@function_tool
async def fetch_user_location(wrapper: RunContextWrapper[UserInfo]) -> str:
    """Returns the location of the user."""
    return f"User {wrapper.context.name} is from {wrapper.context.location}"


async def main():
    user_info = UserInfo(name="Ali", uid=123)

    # Every tool, hook and callback in one run must use the same context type,
    # which is why the agent is written Agent[UserInfo].
    agent = Agent[UserInfo](
        name="Assistant",
        model=llm_model,
        tools=[fetch_user_age, fetch_user_location],
    )

    # The prompt never mentions Ali. The tool reads the name out of wrapper.context.
    result = await Runner.run(
        starting_agent=agent,
        input="What is the age of the user, and where are they from?",
        context=user_info,
    )
    print(result.final_output)

    # The part that surprises people: the wrapper parameter is stripped from the schema,
    # so the model calls fetch_user_age() with no arguments and cannot see or invent the identity.
    print("\nschema the model sees for fetch_user_age:")
    print(fetch_user_age.params_json_schema)


if __name__ == "__main__":
    asyncio.run(main())
