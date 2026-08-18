"""Fixed LCB prompt, run_python tool protocol, and C0/C1/C2 variants."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .data import LCBExample


SYSTEM_BASE = """You are an expert Python competitive-programming agent. Solve the user's problem with a correct and efficient Python 3.11 program using only the standard library. Follow the supplied starter code when present; otherwise read from stdin and write to stdout.

You may call run_python to execute code with optional stdin. Each call is isolated and stateless. Call it at most once per assistant turn and wait for the result.

When the solution is ready, return exactly one fenced Python code block and no other text."""

FINAL_SUBMISSION_REMINDER = """The tool or turn budget is now exhausted. Your next response must be the final submission: exactly one non-empty fenced Python code block and no analysis, explanation, or tool call."""
FINAL_CODE_PREFIX = "```python\n"

PROMPT_STRATEGIES = {
    "c0": "",
    "c1": (
        "Reason concisely about the input/output, constraints, algorithm, and edge cases, "
        "then prioritize submitting the final implementation directly. Use run_python "
        "only for one short, focused check that resolves a specific uncertainty; never "
        "use it to draft the full solution or as a reasoning scratchpad. Reserve most of "
        "the response budget for the final program."
    ),
    "c2": """Follow this workflow:

1. Identify the required input/output behavior, constraints, and important edge cases.
2. Derive a correct algorithm and verify its time and space complexity.
3. Implement the algorithm carefully in the required starter-code or stdin/stdout format.
4. Use run_python to check the provided examples when useful.
5. Check important boundary or adversarial cases, and fix the code if a check fails.
6. Review correctness, complexity, and output formatting before submitting the final code.""",
}

RUN_PYTHON_TOOL = {
    "type": "function",
    "function": {
        "name": "run_python",
        "description": (
            "Execute a short, focused Python 3.11 diagnostic in a fresh isolated "
            "container. Do not use the tool as a scratchpad or include long explanatory "
            "comments. Optional stdin is passed to the program. The call returns exit "
            "status, stdout, stderr, and timeout information. No files or state persist."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Python code to execute."},
                "stdin": {"type": "string", "description": "Optional standard input."},
            },
            "required": ["code"],
        },
    },
}
TOOLS = [RUN_PYTHON_TOOL]

USER_WITH_STARTER = """### Question:
{question}

### Format: You will use the following starter code to write the solution to the problem and enclose your code within delimiters.
```python
{starter_code}
```

### Answer: (use the provided format with backticks)"""

USER_WITHOUT_STARTER = """### Question:
{question}

### Format: Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test on the sample inputs). Enclose your code within delimiters as follows.
```python
# YOUR CODE HERE
```
Ensure that when the python program runs, it reads the inputs, runs the algorithm and writes output to STDOUT.

### Answer: (use the provided format with backticks)"""

TOOL_BLOCK_PATTERN = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
FUNCTION_PATTERN = re.compile(r"<function=([a-zA-Z_][\w]*)>\s*(.*?)\s*</function>", re.DOTALL)
PARAMETER_PATTERN = re.compile(r"<parameter=([a-zA-Z_][\w]*)>\s*(.*?)\s*</parameter>", re.DOTALL)
FINAL_CODE_PATTERN = re.compile(r"\s*```(?:python|py)\s*\n(.*?)```\s*", re.DOTALL | re.IGNORECASE)


@dataclass(frozen=True)
class ParsedAssistant:
    kind: str
    content: str
    code: str | None = None
    stdin: str = ""


def system_prompt(variant: str, *, max_tool_calls: int = 1) -> str:
    normalized = variant.lower()
    if normalized not in PROMPT_STRATEGIES:
        raise ValueError(f"Unknown prompt variant {variant!r}")
    if max_tool_calls < 0:
        raise ValueError("max_tool_calls must be non-negative")
    strategy = PROMPT_STRATEGIES[normalized]
    tool_budget = (
        f"You may call run_python at most {max_tool_calls} times in total. A tool call "
        "must be a short, focused check rather than a full solution or a reasoning "
        "scratchpad; omit long comments and exploratory code. When the tool budget is "
        "exhausted, or when further testing is unnecessary, immediately submit the final "
        "code in the required format."
    )
    parts = [SYSTEM_BASE, tool_budget]
    if strategy:
        parts.append(strategy)
    return "\n\n".join(parts)


def user_prompt(example: LCBExample) -> str:
    if example.starter_code:
        return USER_WITH_STARTER.format(
            question=example.question_content.strip(),
            starter_code=example.starter_code.strip(),
        )
    return USER_WITHOUT_STARTER.format(question=example.question_content.strip())


def initial_messages(
    example: LCBExample,
    variant: str,
    *,
    max_tool_calls: int = 1,
    override_system_prompt: str | None = None,
) -> list[dict[str, Any]]:
    return [
        {
            "role": "system",
            "content": override_system_prompt
            or system_prompt(variant, max_tool_calls=max_tool_calls),
        },
        {"role": "user", "content": user_prompt(example)},
    ]


def _render_chat(tokenizer: Any, messages: list[dict[str, Any]], *, add_generation_prompt: bool) -> list[int]:
    rendered = tokenizer.apply_chat_template(
        messages,
        tools=TOOLS,
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=False,
    )
    if isinstance(rendered, Mapping):
        rendered = rendered["input_ids"]
    if hasattr(rendered, "tolist"):
        rendered = rendered.tolist()
    if rendered and isinstance(rendered[0], list):
        rendered = rendered[0]
    return [int(token) for token in rendered]


def build_prompt(tokenizer: Any, messages: list[dict[str, Any]]) -> list[int]:
    return _render_chat(tokenizer, messages, add_generation_prompt=True)


def encoded_text_tokens(tokenizer: Any, text: str) -> list[int]:
    result = tokenizer.encode(text, add_special_tokens=False)
    if hasattr(result, "tolist"):
        result = result.tolist()
    if result and isinstance(result[0], list):
        result = result[0]
    return [int(token) for token in result]


def _suffix_prefix_overlap(tokens: list[int], suffix: list[int]) -> int:
    for length in range(min(len(tokens), len(suffix)), 0, -1):
        if tokens[-length:] == suffix[:length]:
            return length
    return 0


def build_next_prompt(
    tokenizer: Any,
    messages_before_assistant: list[dict[str, Any]],
    previous_prompt_tokens: list[int],
    completion_tokens: list[int],
    next_tool_message: dict[str, Any],
) -> list[int]:
    canonical_prompt = build_prompt(tokenizer, messages_before_assistant)
    placeholder = {"role": "assistant", "content": "x"}
    with_assistant = [*messages_before_assistant, placeholder]
    assistant_end = _render_chat(tokenizer, with_assistant, add_generation_prompt=False)
    canonical_action = [*canonical_prompt, *encoded_text_tokens(tokenizer, "x")]
    if assistant_end[: len(canonical_action)] != canonical_action:
        raise ValueError("Chat template cannot locate assistant closing boundary")
    closing = assistant_end[len(canonical_action) :]
    next_prompt = build_prompt(tokenizer, [*with_assistant, next_tool_message])
    if next_prompt[: len(assistant_end)] != assistant_end:
        raise ValueError("Chat template rewrote history after tool observation")
    observation = next_prompt[len(assistant_end) :]
    overlap = _suffix_prefix_overlap(completion_tokens, closing)
    return [*previous_prompt_tokens, *completion_tokens, *closing[overlap:], *observation]


def _parse_tool_body(body: str) -> tuple[str, dict[str, Any]] | None:
    function = FUNCTION_PATTERN.fullmatch(body.strip())
    if function:
        return function.group(1), {
            match.group(1): match.group(2).strip()
            for match in PARAMETER_PATTERN.finditer(function.group(2))
        }
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    name = payload.get("name") or payload.get("function")
    arguments = payload.get("arguments", {})
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return None
    return (str(name), dict(arguments)) if name and isinstance(arguments, Mapping) else None


def parse_assistant(text: str) -> ParsedAssistant:
    matches = list(TOOL_BLOCK_PATTERN.finditer(text))
    if matches:
        if len(matches) != 1 or text[matches[0].end() :].strip():
            return ParsedAssistant("invalid", text.strip())
        parsed = _parse_tool_body(matches[0].group(1))
        if parsed is None or parsed[0] != "run_python":
            return ParsedAssistant("invalid", text.strip())
        arguments = parsed[1]
        code = str(arguments.get("code", "")).strip()
        if not code:
            return ParsedAssistant("invalid", text.strip())
        return ParsedAssistant(
            "tool",
            text[: matches[0].start()].strip(),
            code=code,
            stdin=str(arguments.get("stdin", "")),
        )
    if "<tool_call>" in text:
        return ParsedAssistant("invalid", text.strip())
    final = FINAL_CODE_PATTERN.fullmatch(text)
    if final and final.group(1).strip():
        return ParsedAssistant("final", text.strip(), code=final.group(1).strip())
    return ParsedAssistant("invalid", text.strip())


def tool_message(call_id: str, content: str) -> dict[str, Any]:
    return {
        "role": "tool",
        "tool_call_id": call_id,
        "name": "run_python",
        "content": content,
    }


def stop_sequences(tokenizer: Any, *, final_submission: bool = False) -> list[str]:
    eos = getattr(tokenizer, "eos_token", None)
    values = [eos] if eos else []
    if final_submission:
        values.append("```")
    return values
