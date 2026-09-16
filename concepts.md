# Parts 17 and 18, explained simply

Source: openai-agents-sdk-fundamentals-guide-5.pdf, pages 30 to 37.

## Part 17: run lifecycle hooks (RunHooks)

### The five-year-old version

Imagine a school play with several actors. In Part 16 you gave one actor (one agent) a little notebook called `AgentHooks`. Every time that actor walked on stage, used a prop, or walked off, they wrote a line in their notebook.

The problem shows up when the actor hands the scene over to someone else (a handoff, from Part 12). The first actor's notebook goes quiet, because the second actor either has their own notebook or no notebook at all. You lose the story right at the moment it changed hands.

`RunHooks` is a notebook for the whole play instead of one actor. You give it to the director (the `Runner`), and the director writes down what every actor does, from the first line to the last. Handoffs show up in it as their own entry: "this actor left, that actor arrived."

### What it looks like in code

```python
from agents import Agent, Runner, RunHooks, function_tool

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

monitor = SystemMonitor()
result = Runner.run_sync(weather_agent, "What's the latest news on AI coding tools?",
                         hooks=monitor)
print(result.final_output)
print(monitor.timeline)
```

Ask a news question and `monitor.timeline` reads roughly:

```
start WeatherAgent
handoff WeatherAgent -> NewsAgent
start NewsAgent
end NewsAgent
```

One ordered story across both agents. If you attached `AgentHooks` to `WeatherAgent` instead, the trail would stop at the handoff.

Notice where the hooks object goes: `Runner.run_sync(..., hooks=monitor)`. That is the difference from Part 16, where the hooks went on `Agent(..., hooks=...)`.

### The two classes cover the same events with different names

This is the part that bites people. `AgentHooks` and `RunHooks` watch the same moments in the loop, but their method names and arguments do not line up.

| Event | AgentHooks (one agent) | RunHooks (whole run) |
|---|---|---|
| An agent begins working | `on_start(context, agent)` | `on_agent_start(context, agent)` |
| An agent finishes | `on_end(context, agent, output)` | `on_agent_end(context, agent, output)` |
| A handoff happens | `on_handoff(context, agent, source)` | `on_handoff(context, from_agent, to_agent)` |
| Tool starts / ends | `on_tool_start`, `on_tool_end` | same shape |
| LLM call starts / ends | `on_llm_start`, `on_llm_end` | same shape |

Two things to notice.

`on_start` and `on_end` are the per-agent names. `on_agent_start` and `on_agent_end` are the run-level names, and the run-level versions fire once for every agent that takes part. If you override `on_agent_start` on an `AgentHooks` subclass, nothing happens and nothing warns you. Python just sees an extra method the SDK never calls.

`on_handoff` means something different on each side. On `AgentHooks` the arguments are `(context, agent, source)`, where `agent` is the destination and `source` is the sender. On `RunHooks` the point of view is the director: "this one left, that one arrived."

One detail worth knowing from running it (SDK 0.22.2, `run_internal/turn_resolution.py`): the SDK calls `AgentHooks.on_handoff` on the hooks of the agent doing the handing off, not the receiver. So hooks attached to `WeatherAgent` do see the handoff as their final event. What they never see is anything `NewsAgent` does afterwards.

The runner keyword is `hooks=`, the same word used on `Agent(...)`. There is no separate `run_hooks=` argument. The class you pass decides which set of methods gets called.

### When to use which

Reach for `AgentHooks` when the question is about one agent's own behaviour: a specialist you are tuning, or a tool you suspect is slow.

Reach for `RunHooks` when the question is about the request as a whole: total tokens and cost across every agent, an audit log that survives handoffs, or a progress indicator that keeps updating after the conversation changes hands.

They compose. A run can carry a monitor while individual agents carry their own hooks, and both fire.

### Recap of everything in the guide (from page 33)

| Concept | What it gives you |
|---|---|
| `Agent(name, instructions, model, tools)` | Defines who the agent is and what it can use |
| `Runner.run_sync` / `run` / `run_streamed` | Executes the agent against an LLM |
| `asyncio` | Lets LLM calls (network I/O) run without blocking your program |
| Agent / Run / Global model config | Three scopes for choosing which LLM backs an agent |
| `@function_tool` | Turns a Python function into something the agent can call mid-conversation |
| `ModelSettings(temperature, tool_choice, max_tokens)` | Tunes how the chosen model behaves on a run |
| `RunContextWrapper` + `context=` | Passes your own data to tools without showing it to the LLM |
| `instructions=<callable>` | Rebuilds the system prompt on every turn |
| `agent.clone(...)` | Produces a variant of an agent, overriding only named fields |
| `trace(...)` / traces dashboard | Records every step of a run for inspection afterwards |
| `agent.as_tool(...)` | Turns a specialist agent into a tool another agent can call |
| `handoffs=[...]` + `result.last_agent` | Transfers the conversation and tells you who answered |
| `tool_use_behavior`, `max_turns`, `is_enabled` | Control over when the loop stops and which tools exist |
| `output_type=<model>` | Returns a validated Pydantic object instead of prose |
| `@input_guardrail` / `@output_guardrail` | Aborts a run before a bad request is paid for or a bad answer is sent |
| `AgentHooks` | Callbacks at every stage of one agent's work |
| `RunHooks` + `Runner.run(..., hooks=...)` | The same events for every agent in a run, handoffs included |

## Part 18: practice this with Claude Code

### The five-year-old version

Reading a recipe is not the same as cooking. Part 18 says: open a terminal in your project, run `claude`, and ask Claude Code to build each part for you, one prompt at a time. Your job is to read what it wrote, ask "why did you do it that way?", and only move on once you understand the code you now have.

Each prompt builds on the file from the previous one, so the order matters.

### The prompt for each part

| Part | Prompt to give Claude Code |
|---|---|
| 0. Setup | Scaffold a uv-managed Python project called hello-agent with openai-agents and python-dotenv as dependencies. Create a .env file with empty OPENAI_API_KEY and GEMINI_API_KEY entries, and make sure .env is gitignored. |
| 1. OpenAI key | Create main.py using the OpenAI Agents SDK: an Agent named Assistant with instructions "You are a helpful assistant", run synchronously against the prompt "Write a haiku about recursion in programming." Load OPENAI_API_KEY from .env. Then explain what Agent and Runner.run_sync each do. |
| 2. Gemini key | Modify main.py so the same Agent uses Gemini's gemini-2.5-flash model instead of OpenAI's default. Wire up AsyncOpenAI with Gemini's OpenAI-compatible base_url and wrap it in OpenAIChatCompletionsModel. Explain line-by-line why swapping the provider didn't require touching Agent or Runner. |
| 3. Runner and asyncio | Rewrite main.py to call Runner.run inside an async def main() instead of run_sync, executed via asyncio.run(). Then add a second version that fires two different prompts at the same agent concurrently with asyncio.gather and prints both results. Explain why the concurrent version is faster. |
| 4. Model configuration | Refactor my agent so the Gemini model is set at the Run level via RunConfig instead of the Agent level. Show me the same agent being called once with the Gemini run_config and once without one, so I can see the difference. Explain when I'd pick Run-level over Agent-level. |
| 5. Tool use | Add a @function_tool called get_weather(city: str) -> str returning a hardcoded string, and a second tool add(a: int, b: int) -> int. Wire both into the agent's tools list and run it against "What's 42 plus 58, and what's the weather in Lahore?" Then walk me through, step by step, what happened between my prompt and the final answer, including which tools got called and why. |
| 6. Model settings | Take my weather-and-add agent and run the same question three times with model_settings=ModelSettings(tool_choice="auto"), then "required", then "none". Print all three answers side by side and explain what changed and why the "none" answer is different. |
| 7. Local context | Add a UserInfo dataclass with name and uid, pass it to Runner.run via context=, and add a tool that reads wrapper.context.name. Then print the tool's params_json_schema and explain why the wrapper parameter isn't in it. |
| 8. Dynamic instructions | Replace my agent's instructions string with a function taking (context, agent) that greets the user by name from context and mentions the agent's own name. Print the resolved system prompt before the model call, then convert the function into a callable class that counts interactions. |
| 9. Agent cloning | Create a base agent with temperature 0.7 and one tool, then clone it into a creative variant and a precise variant. Prove to me with is-comparisons which attributes are shared between base and clone and which are not. |
| 10. Tracing | Remove set_tracing_disabled from my project, call set_tracing_export_api_key with my OpenAI key, and wrap two consecutive Runner.run calls in a single with trace("...") block. Then tell me exactly which spans I should expect to see on platform.openai.com/traces and in what order. |
| 11. Agents as tools | Build a Writing Coach agent that owns two specialists wrapped with as_tool: a Spanish translator and a summarizer. Run all three of my requests through it, then show me which tool fired for each and explain why the final answer still comes from the coach. |
| 12. Handoffs | Convert my Writing Coach's two as_tool specialists into real handoffs instead. Run the same three requests, print result.last_agent for each, and show me the HandoffCallItem and HandoffOutputItem in result.new_items. Then tell me which of the two designs I should keep and why. |
| 13. Advanced tool control | Add StopAtTools to my agent so the loop ends on a finalizing tool, wrap a run in try/except MaxTurnsExceeded with max_turns=2, and gate one tool behind is_enabled reading a subscription tier from my context. Run it once as a free user and once as premium and show me the difference in the tools the model was offered. |
| 14. Structured output | Give my agent an output_type Pydantic model with three typed fields, run it, and print type(result.final_output) plus one field used in arithmetic to prove it isn't a string. Then feed it a prompt it can't answer in that shape and show me the exception. |
| 15. Guardrails | Add an input guardrail that trips on off-topic requests without calling any model, and catch InputGuardrailTripwireTriggered so my program declines politely instead of crashing. Then prove to me from the trace that no model call was billed for the blocked request. |
| 16. Lifecycle hooks | Attach an AgentHooks subclass that prints on_start, on_tool_start, on_tool_end and on_end, run one question that needs a tool, and show me the order the hooks fired in. Then explain what that order tells me about the agent loop. |
| 17. Run lifecycle hooks | Give my two-agent handoff setup a RunHooks subclass that records on_agent_start, on_handoff and on_agent_end into one list, pass it as hooks= on the run, and print the list. Then attach AgentHooks to the first agent instead and show me where that trail stops. |

### Debug-it prompts (learn by breaking things)

You understand something better once you have had to fix it. Try these after finishing Part 5.

- Deliberately break my Gemini setup by using the wrong base_url, show me the exact error it raises, then walk me through how you diagnosed and fixed it.
- Rename the get_weather function to fetch_weather but leave the docstring and tools list referencing the old name. Run it, show me what breaks, then fix it and explain why the SDK cares about the function name at all.
- Remove the type hint from the city parameter in get_weather. Run the agent again. Does it still work? Explain what the SDK does differently when a tool parameter has no type hint.
- Change my dynamic instructions function so it takes only the context parameter, run it, and show me the exact TypeError the SDK raises before fixing it back.
- Give my two as_tool specialists the same tool_name, run the orchestrator, and show me what happens. Then fix it and explain how the model tells the two tools apart.
- Rename my Fitness Coach agent to Running Coach but leave the router's instructions mentioning transfer_to_fitness_coach. Run it, show me how the routing degrades, and explain where that tool name comes from.
- Override on_agent_start on an AgentHooks subclass instead of on_start, attach it, and run the agent. Show me that nothing prints, then explain which class that method actually belongs to.
- Take my local-context example and remove context=user_info from the Runner.run call, leaving the tool untouched. Run it, show me the exact error, and explain which line of the tool blew up and why.

### Capstone prompt

Once you have been through all eighteen parts, try building something new in one shot, then critique what comes back:

> Using the OpenAI Agents SDK, build a small CLI agent powered by Gemini (Agent-level configuration) with two tools of your own choosing (not get_weather or add). Expose it through an async main() run via asyncio.run(). Then give me three test prompts that demonstrate the agent choosing between the two tools correctly, and explain the tool-selection reasoning for each one.

Compare what Claude Code builds against Parts 1 to 17 of the guide. If you can explain every line without asking "why", you have learned the material.

## Best practices

### For hooks (Parts 16 and 17)

Pick the class by the question you are asking. One agent's behaviour: `AgentHooks` on the agent. The whole request, including handoffs: `RunHooks` on the runner. If you need both, use both; they do not interfere.

Check the method names against the class before you trust silence. `AgentHooks` wants `on_start` / `on_end`. `RunHooks` wants `on_agent_start` / `on_agent_end`. A mismatched override is not an error, it is just a method nobody calls. When a hook "does nothing", this is the first thing to check.

Remember that `on_handoff` has a different signature on each class: `(context, agent, source)` on the agent side, `(context, from_agent, to_agent)` on the run side. Copying a handoff hook from one class to the other will break.

Keep hooks fast. They run inside the agent loop, so a slow database write in `on_tool_end` is time the user spends waiting. Append to a list or push to a queue, and do the heavy work afterwards.

Use hooks for what they are good at: counting tokens and cost per agent, timing which tool is slow, pushing "thinking..." updates to a UI, and writing an audit log of every tool call. For after-the-fact inspection, tracing (Part 10) already does the job without any code in the loop.

Attach a `RunHooks` monitor whenever a run involves handoffs. It is the only view that shows the whole story in order, including the moment the conversation changed hands.

### For learning with Claude Code (Part 18)

Work the prompts in order. Each one edits the file the previous one produced, so skipping ahead leaves you with code that assumes a state you never reached.

Read the diff every time. The guide's rule is to accept nothing you do not understand. Ask "why did you do it that way?" before moving on.

Break things on purpose. The debug prompts exist because a fix you had to find sticks better than a feature you only read about. The Part 17 debug prompt (overriding `on_agent_start` on `AgentHooks`) is a good one to try first since the failure is silent.

Treat the capstone as a test of yourself. The measure is whether you can explain every line of what comes back.
