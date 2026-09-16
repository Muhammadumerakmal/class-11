"""Part 14: Structured output: answers your code can use.

output_type makes the agent return a typed Pydantic object instead of prose. The model
becomes a JSON schema sent alongside the prompt, and the reply is validated against it
before you see it. A reply that doesn't fit raises ModelBehaviorError.

Run:  uv run part14_structured_output.py
"""

from agents import Agent, Runner
from agents.exceptions import ModelBehaviorError
from pydantic import BaseModel

from config import llm_model


class PersonInfo(BaseModel):
    name: str
    age: int
    occupation: str


agent = Agent(
    name="InfoCollector",
    instructions="Extract person information from the user's message.",
    model=llm_model,
    output_type=PersonInfo,
)

message = "Hi, I'm Alice, I'm 25 years old and I work as a teacher."
result = Runner.run_sync(agent, message)
print(type(result.final_output))        # <class 'PersonInfo'>
print(result.final_output.name)         # Alice
print(result.final_output.age + 1)      # 26, a real int, not the text "25"

# When the model returns junk, failure is loud.
print("\n--- a prompt that can't fit the shape ---")
try:
    result = Runner.run_sync(agent, "Tell me a story about a dragon.")
    print("model still produced:", result.final_output)
except ModelBehaviorError as e:
    print("The model didn't produce the shape we asked for:", e)

# If a valid model is rejected by a provider in strict mode, wrap it:
#   from agents import AgentOutputSchema
#   output_type=AgentOutputSchema(PersonInfo, strict_json_schema=False)
