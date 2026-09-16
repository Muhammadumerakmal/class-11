# hello-agent

One runnable script per part of the OpenAI Agents SDK fundamentals guide.

## Setup (Part 0)

```
uv sync
```

Put your keys in `.env` (already gitignored):

```
OPENAI_API_KEY=sk-...
GEMINI_API_KEY=
```

`config.py` builds the shared `llm_model` that Parts 3 to 17 use. If `GEMINI_API_KEY` is set it uses Gemini through the OpenAI-compatible endpoint, as the guide does. If it is empty, `llm_model` is `None` and the SDK falls back to its default OpenAI model, so every script still runs with only an OpenAI key.

## Run a part

```
uv run part01_openai_key.py
uv run part05_tool_use.py
uv run part17_run_hooks.py
```

| File | Part | What it shows |
|---|---|---|
| `part01_openai_key.py` | 1 | Agent, Runner.run_sync, final_output with the default provider |
| `part02_gemini_key.py` | 2 | Swapping the provider with AsyncOpenAI + OpenAIChatCompletionsModel (needs Gemini key) |
| `part03_runner_asyncio.py` | 3 | Runner.run inside async main, two prompts with asyncio.gather |
| `part04_model_config.py` | 4 | Agent, Run and Global level model configuration (needs Gemini key) |
| `part05_tool_use.py` | 5 | @function_tool, one tool then two |
| `part06_model_settings.py` | 6 | temperature, tool_choice auto/required/none, max_tokens |
| `part07_local_context.py` | 7 | context= and RunContextWrapper, params_json_schema is empty |
| `part08_dynamic_instructions.py` | 8 | instructions as a function, then as a stateful callable |
| `part09_cloning.py` | 9 | agent.clone, the shallow-copy trap, settings replaced not merged |
| `part10_tracing.py` | 10 | set_tracing_export_api_key, with trace(...) grouping two runs |
| `part11_agents_as_tools.py` | 11 | as_tool specialists and a function-tool wrapper around Runner.run |
| `part12_handoffs.py` | 12 | handoffs=, last_agent, HandoffCallItem / HandoffOutputItem, follow-up turn |
| `part13_tool_control.py` | 13 | StopAtTools, max_turns, is_enabled gating, tools that return errors |
| `part14_structured_output.py` | 14 | output_type with Pydantic, ModelBehaviorError |
| `part15_guardrails.py` | 15 | keyword input guardrail, agent-based guardrail, output guardrail |
| `part16_agent_hooks.py` | 16 | AgentHooks and the order the callbacks fire |
| `part17_run_hooks.py` | 17 | RunHooks across a handoff, compared with AgentHooks on one agent |

Parts 2 and 4 exit early with a message if `GEMINI_API_KEY` is empty, because their whole point is switching providers.

Part 18 of the guide is the practice workflow: prompt Claude Code for each part, read the diff, and ask why. The prompts are listed in `../concepts.md`.
