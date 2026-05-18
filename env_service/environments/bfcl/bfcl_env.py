# -*- coding: utf-8 -*-
"""
This file is part of https://github.com/ShishirPatil/gorilla

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

# environments/bfcl_env.py
from __future__ import annotations
import ast
import json, os
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List
import re
import uuid

from env_service.base import BaseEnv
from env_service.registry import Registry
from env_service.trajectory import StateMessage, ActionMessage, ToolCall, Role


# 默认路径，可用环境变量覆盖
os.environ.setdefault("BFCL_DATA_PATH", "./bfcl_data/multiturn_data.jsonl")
os.environ.setdefault("BFCL_ANSWER_PATH", "./bfcl_eval/possible_answer")

__all__ = ["BfclEnv"]


T3RL_BFCL_SYSTEM_PROMPT = """You are an expert in composing functions. You are given a question and a set of possible functions. Based on the question, you will need to make one or more function/tool calls to achieve the purpose.

Your response must always start with your step-by-step reasoning process enclosed in `<thinking></thinking>` XML tags. This is for you to outline your plan and justify your chosen actions.

After the thinking block, you will perform one of the following actions:

1. If tool calls are necessary: For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags.
Examples:
<tool_call>\n{"name":"func1", "arguments":{...}}\n</tool_call>\n<tool_call>\n{"name":"func2", "arguments":{...}}\n</tool_call>

2. If no tool calls are necessary or possible: Directly provide a user-facing response in plain text. This applies if none of the functions can be used, or if the given question lacks the parameters required by the function.

At each turn, you should try your best to complete the tasks requested by the user within the current turn. Continue to output functions to call until you have fulfilled the user's request to the best of your ability. Once you have no more functions to call, the system will consider the current turn complete and proceed to the next turn or task."""


LLAMA31_OFFICIAL_SYSTEM_PROMPT = """Environment: ipython
Cutting Knowledge Date: December 2023
Today Date: 26 Jul 2024"""


def _strip_json_code_fence(content: str) -> str:
    text = content.strip()
    if not text.startswith("```"):
        return text
    lines = text.splitlines()
    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _loads_json_or_literal(fragment: str) -> Any:
    try:
        return json.loads(fragment)
    except json.JSONDecodeError:
        return ast.literal_eval(fragment)


def _split_top_level_jsonish_values(text: str) -> List[str]:
    """Split concatenated top-level JSON/Python-literal objects.

    Llama-3.1 often emits multiple raw function-call objects separated by
    newlines when a turn needs several tools. The official prompt prefers a
    JSON list, but accepting newline-separated objects avoids turning a
    semantically good multi-call plan into a parse-error loop.
    """
    fragments: list[str] = []
    stack: list[str] = []
    start: int | None = None
    quote: str | None = None
    escape = False

    matching = {"{": "}", "[": "]"}

    for idx, ch in enumerate(text):
        if start is None:
            if ch.isspace() or ch in ";,":
                continue
            if ch not in matching:
                raise ValueError("expected raw JSON function call")
            start = idx
            stack.append(matching[ch])
            continue

        if quote is not None:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == quote:
                quote = None
            continue

        if ch in ("'", '"'):
            quote = ch
        elif ch in matching:
            stack.append(matching[ch])
        elif ch in ("}", "]"):
            if not stack or ch != stack[-1]:
                raise ValueError("mismatched JSON brackets")
            stack.pop()
            if not stack:
                fragments.append(text[start : idx + 1].strip())
                start = None

    if start is not None or stack:
        raise ValueError("incomplete raw JSON function call")
    if not fragments:
        raise ValueError("expected raw JSON function call")
    return fragments


def _parse_llama31_payload(content: str) -> Any:
    text = _strip_json_code_fence(content.replace("<|python_tag|>", "")).strip()
    if not text:
        raise ValueError("empty response")
    try:
        return _loads_json_or_literal(text)
    except Exception:
        pass

    fragments = _split_top_level_jsonish_values(text)
    if len(fragments) > 1:
        return [_loads_json_or_literal(fragment) for fragment in fragments]

    raise ValueError("expected raw JSON function call")


def parse_llama31_official_content_to_tool_calls(
    content: str, strict: bool = False
) -> Dict[str, Any]:
    stripped = content.strip()
    if not stripped:
        return {"role": "assistant", "content": "", "tool_calls": []}

    has_tool_tag = "<tool_call" in stripped or "</tool_call>" in stripped
    looks_structured = stripped.startswith("{") or stripped.startswith("[") or stripped.startswith("```")
    if has_tool_tag:
        result = {"role": "assistant", "content": stripped, "tool_calls": []}
        if strict:
            result["_bfcl_parse_error"] = (
                "Llama-3.1 official mode expects raw JSON, not <tool_call> XML tags."
            )
        return result

    try:
        payload = _parse_llama31_payload(stripped)
    except Exception as exc:
        result = {"role": "assistant", "content": stripped, "tool_calls": []}
        if strict and looks_structured:
            result["_bfcl_parse_error"] = f"Invalid Llama-3.1 JSON function call: {exc}"
        return result

    entries = payload if isinstance(payload, list) else [payload]
    tool_calls: list[dict[str, Any]] = []
    parse_errors: list[str] = []
    call_id_counter = 1
    for entry in entries:
        if not isinstance(entry, dict):
            parse_errors.append("Function call entry must be a JSON object.")
            continue
        func_name = entry.get("name") or entry.get("function")
        arguments = entry.get("parameters", entry.get("arguments"))
        if not func_name:
            parse_errors.append("Function call JSON must contain a 'name' key.")
            continue
        if arguments is None:
            parse_errors.append("Function call JSON must contain a 'parameters' object.")
            continue
        if isinstance(arguments, str):
            try:
                arguments = _loads_json_or_literal(arguments)
            except Exception:
                parse_errors.append(f"Tool '{func_name}' has non-JSON string parameters.")
                continue
        if strict and not isinstance(arguments, dict):
            parse_errors.append(f"Tool '{func_name}' parameters must be a JSON object.")
            continue
        tool_calls.append(
            {
                "id": f"{func_name}_{call_id_counter}",
                "type": "function",
                "function": {
                    "name": str(func_name),
                    "arguments": arguments,
                },
            }
        )
        call_id_counter += 1

    result = {"role": "assistant", "content": "", "tool_calls": tool_calls}
    if strict and parse_errors:
        result["_bfcl_parse_error"] = "; ".join(parse_errors)
    return result


def parse_assistant_content_to_tool_calls(
    msg: Dict[str, Any],
    strict: bool = False,
    parser_mode: str = "xml_json",
    reject_tool_call_with_content: bool = False,
    flag_python_literals: bool = False,
    accept_parameters_as_arguments: bool = False,
) -> Dict[str, Any]:
    """
    从 assistant 的 content 中解析出 tool_calls，并返回新的消息结构。
    支持 Qwen 的 <tool_call>...<tool_call> 工具调用格式。

    Args:
        msg (dict): 原始 assistant 消息，包含 'content' 字段

    Returns:
        dict: 包含 'content' 和 'tool_calls' 的新消息结构
    """
    content = msg.get("content", "") or ""
    if not isinstance(content, str):
        content = str(content)

    if parser_mode == "llama31_official_fc":
        return parse_llama31_official_content_to_tool_calls(content, strict=strict)
    if parser_mode in {"toolace_fc", "toolace_official_prompt"}:
        return parse_toolace_content_to_tool_calls(
            content,
            strict=strict,
            reject_tool_call_with_content=reject_tool_call_with_content,
        )
    if parser_mode == "envtuning_fc":
        return parse_envtuning_content_to_tool_calls(content, strict=strict)

    tool_calls = []
    remaining_content = content
    call_id_counter = 1

    # 正则匹配 <tool_call> ... asdf ... asdf ...
    pattern = r'<tool_call>\s*\n?({.*?})\s*\n?\</tool_call>'
    matches = list(re.finditer(pattern, content, re.DOTALL))
    parse_errors: list[str] = []
    has_tool_tag = "<tool_call" in content or "</tool_call>" in content

    if not matches:
        result = {
            "role": "assistant",
            "content": content.strip(),
            "tool_calls": []
        }
        if strict and has_tool_tag:
            result["_bfcl_parse_error"] = (
                "Found tool_call tag text, but no valid <tool_call>{...}</tool_call> block."
            )
        return result

    # 提取所有匹配的 JSON 字符串
    for match in matches:
        json_str = match.group(1).strip()
        try:
            data = json.loads(json_str)
            if not isinstance(data, dict):
                continue
            if (
                accept_parameters_as_arguments
                and "arguments" not in data
                and "parameters" in data
            ):
                data["arguments"] = data["parameters"]
            if "name" not in data or "arguments" not in data:
                parse_errors.append("Tool call JSON must contain both 'name' and 'arguments'.")
                continue
            
            func_name = data["name"]
            arguments = data["arguments"]
            if strict:
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        parse_errors.append(
                            f"Tool '{func_name}' has non-JSON string arguments."
                        )
                        continue
                if not isinstance(arguments, dict):
                    parse_errors.append(
                        f"Tool '{func_name}' arguments must be a JSON object."
                    )
                    continue
            tool_call = {
                "id": f"{func_name}_{call_id_counter}",
                "type": "function",
                "function": {
                    "name": data["name"],
                    "arguments": arguments
                }
            }
            tool_calls.append(tool_call)
            call_id_counter += 1
        except json.JSONDecodeError as e:
            print(f"JSON 解析失败: {json_str[:50]}... -> {e}")
            literal_hint = ""
            if flag_python_literals and re.search(r"\b(True|False|None)\b", json_str):
                literal_hint = (
                    " (use JSON true/false/null, not Python True/False/None)"
                )
            parse_errors.append(f"Invalid tool call JSON{literal_hint}: {e}")
            continue

    # 移除所有 tool call 部分，得到纯文本 content
    cleaned_content = re.sub(pattern, '', content, flags=re.DOTALL).strip()
    # 可选：清理多余的空白
    cleaned_content = re.sub(r'\n\s*\n', '\n\n', cleaned_content).strip()
    if strict and reject_tool_call_with_content and cleaned_content:
        parse_errors.append(
            "Tool-call output must not include text outside <tool_call> blocks."
        )

    result = {
        "role": "assistant",
        "content": cleaned_content,
        "tool_calls": tool_calls
    }
    if strict and parse_errors:
        result["_bfcl_parse_error"] = "; ".join(parse_errors)

    return result


def _extract_single_xml_block(content: str, tag: str) -> tuple[str | None, list[str]]:
    pattern = rf"<{tag}>([\s\S]*?)</{tag}>"
    matches = re.findall(pattern, content, flags=re.DOTALL)
    if not matches:
        return None, []
    return matches[0].strip(), matches


def parse_envtuning_content_to_tool_calls(
    content: str, strict: bool = False
) -> Dict[str, Any]:
    """Parse EnvTuning-style BFCL responses.

    EnvTuning's multi-turn FC interaction asks the model to emit exactly one
    thinking block and then either a JSON tool payload inside <tool_call> tags
    or a final answer inside <answer> tags. This mode is kept separate from the
    BFCL ToolACE prompt-mode parser because it is an interaction protocol, not
    the official prompt-mode string parser.
    """
    raw = content.strip()
    think_body, think_matches = _extract_single_xml_block(raw, "think")
    tool_body, tool_matches = _extract_single_xml_block(raw, "tool_call")
    answer_body, answer_matches = _extract_single_xml_block(raw, "answer")

    parse_errors: list[str] = []
    if strict:
        if not think_matches:
            parse_errors.append("Missing <think></think> block.")
        if len(think_matches) > 1:
            parse_errors.append("Multiple <think></think> blocks found.")
        if len(tool_matches) > 1:
            parse_errors.append("Multiple <tool_call></tool_call> blocks found.")
        if len(answer_matches) > 1:
            parse_errors.append("Multiple <answer></answer> blocks found.")
        if tool_matches and answer_matches:
            parse_errors.append("Use either <tool_call> or <answer>, not both.")

    if tool_body is None:
        if answer_body is not None:
            result = {"role": "assistant", "content": answer_body, "tool_calls": []}
        else:
            result = {"role": "assistant", "content": raw, "tool_calls": []}
            if strict and ("<tool_call" in raw or "</tool_call>" in raw):
                parse_errors.append("Malformed <tool_call> block.")
        if strict and parse_errors:
            result["_bfcl_parse_error"] = "; ".join(parse_errors)
        return result

    try:
        payload = _loads_json_or_literal(_strip_json_code_fence(tool_body))
    except Exception as exc:
        result = {"role": "assistant", "content": raw, "tool_calls": []}
        if strict:
            parse_errors.append(f"Invalid JSON inside <tool_call>: {exc}")
            result["_bfcl_parse_error"] = "; ".join(parse_errors)
        return result

    entries = payload if isinstance(payload, list) else [payload]
    tool_calls: list[dict[str, Any]] = []
    call_id_counter = 1
    for entry in entries:
        if not isinstance(entry, dict):
            parse_errors.append("Tool call entry must be a JSON object.")
            continue
        func_name = entry.get("name") or entry.get("function")
        arguments = entry.get("arguments", entry.get("parameters", {}))
        if not isinstance(func_name, str) or not func_name.strip():
            parse_errors.append("Tool call JSON must contain a string 'name'.")
            continue
        if isinstance(arguments, str):
            try:
                arguments = _loads_json_or_literal(arguments)
            except Exception:
                parse_errors.append(f"Tool '{func_name}' has non-JSON string arguments.")
                arguments = {}
        if arguments is None:
            arguments = {}
        if not isinstance(arguments, dict):
            parse_errors.append(f"Tool '{func_name}' arguments must be a JSON object.")
            arguments = {}
        tool_calls.append(
            {
                "id": f"{func_name}_{call_id_counter}",
                "type": "function",
                "function": {
                    "name": func_name.strip(),
                    "arguments": arguments,
                },
            }
        )
        call_id_counter += 1

    result = {"role": "assistant", "content": think_body or "", "tool_calls": tool_calls}
    if strict and parse_errors:
        result["_bfcl_parse_error"] = "; ".join(parse_errors)
    return result


def _toolace_ast_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _toolace_ast_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return None


def _toolace_literal(node: ast.AST) -> Any:
    try:
        return ast.literal_eval(node)
    except Exception:
        try:
            return ast.unparse(node)
        except Exception:
            return None


def _loads_toolace_jsonish_payload(content: str) -> Any:
    raw = _strip_json_code_fence(content).strip()
    candidates = [raw]
    if raw.startswith("{") and re.search(r"}\s*[,;]\s*{", raw):
        joined = re.sub(r"}\s*;\s*{", "}, {", raw)
        candidates.insert(0, f"[{joined}]")

    last_error: Exception | None = None
    for candidate in candidates:
        text = re.sub(
            r'((?:"name"|"function")\s*:\s*)([A-Za-z_][\w.]*)',
            lambda m: f'{m.group(1)}"{m.group(2)}"',
            candidate,
        )
        try:
            return _loads_json_or_literal(text)
        except Exception as exc:
            last_error = exc
        try:
            pythonish = re.sub(r"\btrue\b", "True", text)
            pythonish = re.sub(r"\bfalse\b", "False", pythonish)
            pythonish = re.sub(r"\bnull\b", "None", pythonish)
            return ast.literal_eval(pythonish)
        except Exception as exc:
            last_error = exc

    if last_error is not None:
        raise last_error
    raise ValueError("empty ToolACE JSON-style payload")


def _parse_toolace_jsonish_to_tool_calls(
    content: str, strict: bool = False
) -> Dict[str, Any] | None:
    stripped = content.strip()
    looks_jsonish = stripped.startswith("{") or stripped.startswith("[") or stripped.startswith("```")
    if not looks_jsonish:
        return None

    try:
        payload = _loads_toolace_jsonish_payload(stripped)
    except Exception as exc:
        result = {"role": "assistant", "content": content.strip(), "tool_calls": []}
        if strict:
            result["_bfcl_parse_error"] = f"Invalid ToolACE JSON-style function call: {exc}"
        return result

    entries = payload if isinstance(payload, list) else [payload]
    tool_calls: list[dict[str, Any]] = []
    parse_errors: list[str] = []
    call_id_counter = 1
    for entry in entries:
        if not isinstance(entry, dict):
            parse_errors.append("ToolACE JSON-style call entry must be an object.")
            continue
        func_name = entry.get("name") or entry.get("function")
        if not isinstance(func_name, str) or not func_name.strip():
            parse_errors.append("ToolACE JSON-style call must contain a string 'name'.")
            continue
        arguments = entry.get("parameters", entry.get("arguments", {}))
        if arguments is None:
            arguments = {}
        if isinstance(arguments, str):
            try:
                arguments = _loads_json_or_literal(arguments)
            except Exception:
                parse_errors.append(f"Tool '{func_name}' has non-JSON string parameters.")
                arguments = {}
        if not isinstance(arguments, dict):
            parse_errors.append(f"Tool '{func_name}' parameters must be an object.")
            arguments = {}
        tool_calls.append(
            {
                "id": f"{func_name}_{call_id_counter}",
                "type": "function",
                "function": {
                    "name": func_name.strip(),
                    "arguments": arguments,
                },
            }
        )
        call_id_counter += 1

    result = {"role": "assistant", "content": "", "tool_calls": tool_calls}
    if strict and parse_errors:
        result["_bfcl_parse_error"] = "; ".join(parse_errors)
    return result


def _toolace_bracket_spans(content: str) -> list[tuple[int, int, str]]:
    spans: list[tuple[int, int, str]] = []
    depth = 0
    start: int | None = None
    in_string: str | None = None
    escape = False
    for idx, ch in enumerate(content):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == in_string:
                in_string = None
            continue
        if ch in {"'", '"'}:
            in_string = ch
            continue
        if ch == "[":
            if depth == 0:
                start = idx
            depth += 1
        elif ch == "]" and depth > 0:
            depth -= 1
            if depth == 0 and start is not None:
                snippet = content[start : idx + 1]
                if "(" in snippet and ")" in snippet:
                    spans.append((start, idx + 1, snippet))
                start = None
    return spans


def parse_toolace_content_to_tool_calls(
    content: str,
    strict: bool = False,
    reject_tool_call_with_content: bool = False,
) -> Dict[str, Any]:
    spans = _toolace_bracket_spans(content)
    if not spans:
        stripped = content.strip()
        if "<tool_call" in stripped or "</tool_call>" in stripped:
            result = {"role": "assistant", "content": stripped, "tool_calls": []}
            if strict:
                result["_bfcl_parse_error"] = (
                    "ToolACE BFCL prompt mode expects a Python function-call list, "
                    'e.g. [search(query="value")], not <tool_call> tags.'
                )
            return result
        if (
            re.match(r"^[A-Za-z_][\w.]*\s*\(", stripped)
            and stripped.endswith(")")
        ):
            spans = [(0, len(content), stripped)]
        else:
            jsonish_result = _parse_toolace_jsonish_to_tool_calls(content, strict=strict)
            if jsonish_result is not None:
                return jsonish_result
            if stripped.startswith("[") or stripped.endswith("]"):
                return {
                    "role": "assistant",
                    "content": content.strip(),
                    "tool_calls": [],
                    "_bfcl_parse_error": "Invalid ToolACE function-call list.",
                }
            return {"role": "assistant", "content": content.strip(), "tool_calls": []}

    tool_calls: list[dict[str, Any]] = []
    parse_errors: list[str] = []
    call_id_counter = 1
    consumed: list[tuple[int, int]] = []

    for start, end, snippet in spans:
        try:
            parsed = ast.parse(snippet, mode="eval")
        except SyntaxError as e:
            parse_errors.append(f"Invalid ToolACE function-call syntax: {e.msg}")
            continue
        body = parsed.body
        if isinstance(body, (ast.List, ast.Tuple)):
            elements = body.elts
        elif isinstance(body, ast.Call):
            elements = [body]
        else:
            parse_errors.append("ToolACE output must be a list of function calls.")
            continue
        consumed.append((start, end))
        for call in elements:
            if not isinstance(call, ast.Call):
                parse_errors.append("ToolACE list contains a non-call item.")
                continue
            func_name = _toolace_ast_name(call.func)
            if not func_name:
                parse_errors.append("ToolACE call has an invalid function name.")
                continue
            if strict and call.args:
                parse_errors.append(f"Tool '{func_name}' uses positional arguments.")
                continue
            arguments: dict[str, Any] = {}
            for kw in call.keywords:
                if kw.arg is None:
                    parse_errors.append(f"Tool '{func_name}' uses unsupported **kwargs.")
                    continue
                arguments[kw.arg] = _toolace_literal(kw.value)
            tool_calls.append(
                {
                    "id": f"{func_name}_{call_id_counter}",
                    "type": "function",
                    "function": {
                        "name": func_name,
                        "arguments": arguments,
                    },
                }
            )
            call_id_counter += 1

    cleaned = content
    for start, end in reversed(consumed):
        cleaned = cleaned[:start] + cleaned[end:]
    cleaned = re.sub(r"\n\s*\n", "\n\n", cleaned).strip()
    if strict and reject_tool_call_with_content and tool_calls and cleaned:
        parse_errors.append(
            "ToolACE output must not include text outside the function-call list."
        )

    result = {"role": "assistant", "content": cleaned, "tool_calls": tool_calls}
    if strict and parse_errors:
        result["_bfcl_parse_error"] = "; ".join(parse_errors)
    return result


def tools_schema_to_llama31_official_prompt(
    tools_schema: List[Dict[str, Any]],
    user_query: str,
    extra_instructions: List[str] | None = None,
) -> str:
    lines = [
        "Given the following functions, please respond with a JSON for a function call "
        "with its proper arguments that best answers the given prompt.",
        "",
        'For one function call, respond in the format {"name": function name, "parameters": dictionary of argument name and its value}.',
        'If multiple function calls are needed in the same turn, respond with one raw JSON list of those objects, for example [{"name": "func1", "parameters": {...}}, {"name": "func2", "parameters": {...}}].',
        "Do not use variables.",
        "Return only raw JSON for function calls; do not wrap the JSON in <tool_call> tags.",
        "Do not emit several standalone JSON objects on separate lines; use a JSON list for multiple calls.",
        "If no function can be used, or required parameters are missing, answer briefly in plain text.",
        "",
    ]
    for tool in tools_schema:
        lines.append(json.dumps(tool, ensure_ascii=False, indent=4))
        lines.append("")
    extras = [str(item).strip() for item in (extra_instructions or []) if str(item).strip()]
    if extras:
        lines.extend(extras)
        lines.append("")
    lines.append(user_query.strip())
    return "\n".join(lines).strip()


def tools_schema_to_toolace_official_prompt(
    tools_schema: List[Dict[str, Any]],
) -> str:
    """BFCL official prompt-mode interface used for ToolACE/watt Llama models.

    ToolACE/watt expects the key "arguments" (not "parameters") in each function
    definition.  BFCL's internal schema uses "parameters", so we rename it here
    before serialising to the system-prompt.
    """
    converted = []
    for tool in tools_schema:
        # Support both bare-function dicts and OpenAI {"type": "function", "function": {...}} wrappers.
        func = tool.get("function", tool)
        entry: Dict[str, Any] = {
            "name": func.get("name", ""),
            "description": func.get("description", ""),
        }
        # Rename "parameters" → "arguments" as expected by ToolACE/watt models.
        params = func.get("parameters") or func.get("arguments")
        if params is not None:
            entry["arguments"] = params
        converted.append(entry)
    function_docs = json.dumps(converted, ensure_ascii=False, indent=4)
    return (
        "You are an expert in composing functions. You are given a question and a "
        "set of possible functions. Based on the question, you will need to make "
        "one or more function/tool calls to achieve the purpose.\n"
        "If none of the functions can be used, point it out. If the given question "
        "lacks the parameters required by the function, also point it out.\n"
        "You should only return the function calls in your response.\n\n"
        "If you decide to invoke any of the function(s), you MUST put it in the "
        "format of [func_name1(params_name1=params_value1, "
        "params_name2=params_value2...), func_name2(params)].\n"
        "You SHOULD NOT include any other text in the response.\n\n"
        "Here is a list of functions in JSON format that you can invoke.\n"
        f"{function_docs}\n"
    )


def tools_schema_to_envtuning_prompt(
    tools_schema: List[Dict[str, Any]],
) -> str:
    """EnvTuning-compatible XML + JSON interaction protocol for BFCL."""
    function_docs = json.dumps(tools_schema, ensure_ascii=False, separators=(",", ":"))
    return (
        "You are an expert in composing functions. You are given a question and "
        "a set of possible functions. Based on the question, you will need to "
        "make one or more function/tool calls to achieve the purpose.\n\n"
        "Your response must first outline your thought process inside exactly "
        "one <think></think> pair.\n\n"
        "If you decide to invoke function(s), return a JSON object or JSON list "
        "inside exactly one <tool_call></tool_call> pair. Each object must use "
        'the schema {"name": <function-name>, "arguments": <args-json-object>}.\n'
        "If none of the functions can be used, or if the request lacks required "
        "parameters, provide the final response inside exactly one "
        "<answer></answer> pair.\n\n"
        "Do not include text outside the XML tags. Do not use Python-style "
        "function-call syntax in this mode.\n\n"
        "Here is a list of functions in JSON format that you can invoke:\n"
        "<tools>\n"
        f"{function_docs}\n"
        "</tools>"
    )


def tools_schema_to_qwen_prompt(tools_schema, prompt_mode: str = "bfcl_qwen_fc"):
    """
    将 tools_schema 转换为符合 Qwen 模型 chat_template 的工具描述 prompt。

    Args:
        tools_schema (list): 工具列表，格式如下：
            [
                {
                    "name": "tool_name",
                    "description": "工具描述",
                    "parameters": {
                        "type": "object",
                        "properties": { ... },
                        "required": [ ... ]
                    }
                }
            ]

    Returns:
        str: 包含 <tools> 标签的完整 system 工具描述 prompt
    """
    if prompt_mode == "toolace_official_prompt":
        return tools_schema_to_toolace_official_prompt(tools_schema)
    if prompt_mode == "envtuning_fc":
        return tools_schema_to_envtuning_prompt(tools_schema)

    if not tools_schema:
        return ""

    lines = []
    if prompt_mode == "bfcl_qwen_fc":
        lines.append("# Tools\n")
        lines.append("You may call one or more functions to assist with the user query.\n")
        lines.append("You are provided with function signatures within <tools></tools> XML tags:")
    elif prompt_mode == "toolace_fc":
        lines.append("You are an expert in composing functions. You are given a question and a set of possible functions. Based on the question, you will need to make one or more function/tool calls to achieve the purpose.")
        lines.append("If none of the functions can be used, point it out. If the given question lacks the parameters required by the function, also point it out.")
        lines.append("You should only return the function call in tools call sections.")
        lines.append("If you decide to invoke any function(s), you MUST put them in the format of [func_name1(param_name1=param_value1, param_name2=param_value2), func_name2(param=value)].")
        lines.append("You SHOULD NOT include any other text in the response.")
        lines.append("Here is a list of functions in JSON format that you can invoke:")
    elif prompt_mode == "t3rl_text":
        lines.append(T3RL_BFCL_SYSTEM_PROMPT.strip())
        lines.append("\n\n# Tools\n")
        lines.append("You are provided with function signatures within <tools></tools> XML tags:")
    else:
        lines.append("\n\n# Tools\n")
        lines.append("You may call one or more functions to assist with the user query.\n")
        lines.append("You are provided with function signatures within <tools></tools> XML tags:")
    lines.append("<tools>")
    for tool in tools_schema:
        if prompt_mode == "bfcl_qwen_fc":
            # Match BFCL's QwenFCHandler prompt style as closely as possible.
            tool_json = json.dumps(tool)
        elif prompt_mode == "toolace_fc":
            func = tool.get("function", tool)
            toolace_tool = {
                "name": func.get("name"),
                "description": func.get("description", ""),
                "arguments": func.get("parameters", {}),
            }
            tool_json = json.dumps(
                toolace_tool,
                ensure_ascii=False,
                separators=(',', ':')
            )
        else:
            tool_json = json.dumps(
                tool,
                ensure_ascii=False,
                separators=(',', ':')
            )
        lines.append(tool_json)
    lines.append("</tools>\n")
    if prompt_mode == "bfcl_qwen_fc":
        lines.append("For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:")
        lines.append("<tool_call>")
        lines.append('{"name": <function-name>, "arguments": <args-json-object>}')
        lines.append("</tool_call>")
    elif prompt_mode == "toolace_fc":
        lines.append("Remember: output only a bracketed function-call list, e.g. [search(query=\"value\")].")
    elif prompt_mode != "t3rl_text":
        lines.append("Important: Always use only the latest tool list provided, ignoring any functions mentioned in previous messages.")
        lines.append("For each function call, return a json object with function name and arguments within <tool_call> and <tool_call> XML tags:")
        lines.append("<tool_call>")
        lines.append('{\"name\": <function-name>, \"arguments\": <args-json-object>}')
        lines.append("</tool_call>")

    return "\n".join(lines)

def tool_message_to_qwen_text(tool_messages, result_mode: str = "bfcl_tool_response"):
    """
    将 role 为 'tool' 的消息列表转换为符合 Qwen chat_template 格式的字符串。
    支持单个或多个连续的 tool 消息。

    Args:
        tool_messages (list or dict): 一个或多个 tool 消息字典

    Returns:
        str: 符合 Qwen 模板的文本表示，包含 <|im_start|>user ... <|im_end|>
    """
    if isinstance(tool_messages, dict):
        tool_messages = [tool_messages]

    if not tool_messages:
        return ""

    # AgentEvolver feeds tool observations back as user-role text. The
    # bfcl_tool_response branch mirrors BFCL QwenFCHandler's tool rendering.
    tool_entries = []
    for msg in tool_messages:
        if msg.get("role") != "tool":
            raise ValueError("All messages must have role 'tool'")

        content = msg.get("content", "")
        tool_call_id = msg.get("tool_call_id", "")
        # NOTICE: yunpeng - bfcl 不返回toolname，用id代替
        name = msg.get("name", tool_call_id)  # 工具名称 

        if not name:
            raise ValueError("Missing 'name' in tool message.")

        if result_mode == "bfcl_tool_response":
            if isinstance(content, str):
                content_text = content
            else:
                content_text = json.dumps(content, ensure_ascii=False)
            tool_entries.append(f"<tool_response>\n{content_text}\n</tool_response>")
            continue
        if result_mode == "llama31_official":
            if isinstance(content, str):
                content_text = content
            else:
                content_text = json.dumps(content, ensure_ascii=False)
            tool_entries.append(content_text)
            continue
        if result_mode == "toolace_official_prompt":
            # vLLM's ToolACE chat template renders tool/ipython messages as a
            # JSON object.  For string observations it emits
            # {"output": message.content}; for mapping observations it emits the
            # mapping directly.  Matching that surface keeps Llama/ToolACE from
            # seeing a quoted JSON string as the whole tool result.
            if isinstance(content, dict):
                tool_entries.append(json.dumps(content, ensure_ascii=False))
            else:
                tool_entries.append(
                    json.dumps({"output": str(content)}, ensure_ascii=False)
                )
            continue
        if result_mode == "envtuning_fc":
            if isinstance(content, str):
                try:
                    content_obj = json.loads(content)
                except Exception:
                    content_obj = content
            else:
                content_obj = content
            response_content = json.dumps(content_obj, ensure_ascii=False)
            tool_entries.append(
                "Here are the function's execution results. "
                f"Execution results:{response_content}\n "
                "If you believe you have already fulfilled the user's request, "
                "please first outline your thought process in a <think></think> "
                "pair, and then give a brief summary of the result in an "
                "<answer></answer> pair. Otherwise, you should continue to call "
                "until fulfilling user's request."
            )
            continue

        # 确保 content 是 JSON 可序列化对象
        try:
            if isinstance(content, str):
                parsed_content = json.loads(content) if content.strip().startswith(('{', '[')) else content
            else:
                parsed_content = content
        except Exception:
            parsed_content = content

        # 构造工具返回的标准结构：{"name": "...", "content": ...}
        entry = {
            "name": name,
            "content": parsed_content
        }
        if result_mode == "plain_user":
            if isinstance(parsed_content, str):
                content_text = parsed_content
            else:
                content_text = json.dumps(parsed_content, ensure_ascii=False)
            tool_entries.append(f"Tool result from {name}:\n{content_text}")
        else:
            tool_entries.append(f'<tool_call>\n{json.dumps(entry, ensure_ascii=False)}\n</tool_call>')

    # 合并所有 tool entry，用换行连接
    inner_text = "\n".join(tool_entries) + "\n"

    return inner_text

@Registry.register("bfcl")
class BfclEnv(BaseEnv):
    """Berkeley-Function-Calling-Leaderboard 多轮对话环境"""

    # ------------------------------------------------------------------ #
    # 初始化
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        task_id: str | None = None,
        instance_id: str | None = None,
        params: Dict[str, Any] | None = None,
    ):
        self.task_id, self.instance_id = task_id, instance_id
        self.params: Dict[str, Any] = params or {}

        self.data_path = self.params.get("data_path", os.getenv("BFCL_DATA_PATH"))
        self.answer_path = self.params.get("answer_path", os.getenv("BFCL_ANSWER_PATH"))
        self.model_name = self.params.get("model_name", "env_handler")

        # runtime
        self.test_entry: Dict[str, Any] | None = None
        self.original_test_entry: Dict[str, Any] | None = None
        self.env_handler = None
        self.conversation_history: list[Dict[str, Any]] = []
        self.current_turn = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.tools_info = ""

    # ------------------------------------------------------------------ #
    # 生命周期
    # ------------------------------------------------------------------ #
    def get_init_state(self, params: Dict[str, Any] | None = None) -> Dict[str, Any]:
        """载入测试用例并返回首条 user 消息"""
        if params:
            self.params.update(params)
        self.test_entry = self._load_test_case(self.data_path, self.task_id)
        self.original_test_entry = self.test_entry

        # 必须成功实例化真实 EnvHandler
        from env_service.environments.bfcl.env_handler import EnvHandler

        self.env_handler = EnvHandler(
            model_name=self.model_name, answer_path=Path(self.answer_path)
        )

        # 初始历史
        self.conversation_history = self.test_entry.get("question", [[]])[0].copy()
        self.current_turn = 0

        # 工具信息
        tools = self.test_entry.get("function", [])
        presented_tools = self._apply_observation_lite(tools)
        presented_tools = self._apply_diagnostic_evolution(presented_tools)
        self._cached_tool_names = [
            str((t.get("function", t) or {}).get("name") or "")
            for t in presented_tools
            if (t.get("function", t) or {}).get("name")
        ]
        # print("tools:", tools)
        self.tools_info = "Available tools:\n" + "\n".join(
            f"- {t.get('function', {}).get('name', 'unknown')}" for t in tools
        )

        first_query = (
            self.conversation_history[0]["content"] if self.conversation_history else ""
        )

        # add system prompt: czy-0709
        # from bfcl_eval.constants.default_prompts import DEFAULT_SYSTEM_PROMPT
        # from bfcl_eval.model_handler.utils import func_doc_language_specific_pre_processing
        # system_prompt_template = DEFAULT_SYSTEM_PROMPT
        # functions = self.original_test_entry["function"]
        # test_category = self.test_entry["id"].rsplit("_", 1)[0]
        # function_docs = func_doc_language_specific_pre_processing(functions, test_category)
        # system_prompt = system_prompt_template.format(functions=function_docs)
        prompt_mode = str(self.params.get("tool_prompt_mode", "bfcl_qwen_fc"))
        extra_instructions: list[str] = []
        if self._observation_lite_enabled():
            extra_instructions.append(self._observation_lite_instruction())
        diag_block = self._diagnostic_evolution_system_block()
        if diag_block:
            extra_instructions.append(diag_block)
        if prompt_mode == "llama31_official_fc":
            system_content = LLAMA31_OFFICIAL_SYSTEM_PROMPT
            user_content = tools_schema_to_llama31_official_prompt(
                presented_tools,
                first_query,
                extra_instructions=extra_instructions,
            )
        else:
            system_content = tools_schema_to_qwen_prompt(
                presented_tools,
                prompt_mode=prompt_mode,
            )
            if extra_instructions:
                system_content = f"{system_content}\n\n" + "\n\n".join(extra_instructions)
            user_content = first_query
        return {
            # system_prompt + "\n\n" + first_query
            "state": [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content}
                ],
            "info": {
                "instance_id": self.instance_id,
                "task_id": self.task_id,
                "test_id": self.test_entry.get("id", "unknown"),
                "tools_count": len(tools),
                "questions_count": len(self.original_test_entry.get("question", [])),
                "observation_lite": bool(self._observation_lite_enabled()),
            },
        }

    def step(
        self, action: Dict[str, Any], params: Dict[str, Any] | None = None
    ) -> Dict[str, Any]:
        # action: {'content': '<think>\n\n</think>\n\n', 'role': 'assistant', 'tool_calls': [{'id': 'chatcmpl-tool-xxx', 'function': {'arguments': '{..}', 'name': '...'}, 'type': 'function'},{...}]
        ### change by czy0712
        # action_msg = ActionMessage(**action)
        cur_turn=self.current_turn
        self._last_step_info = {}
        state_msg = self.transition(action, params=params or {}) # change by czy0712
        # state_msg: role=<Role.USER: 'user'> content='' reasoning_content='' tool_calls=[ToolCall(...)] timestamp='2025-xxx' metadata={} tool_call_id=''
        terminated = self._is_terminated(state_msg.simple_dict["content"]) # change by czy0721
        # if new query is ready to be sent but single_turn is True
        if cur_turn!=self.current_turn and (self.params.get('is_open_query',False)==True):
            # terminate the trajectory
            terminated=True
        step_info = dict(getattr(self, "_last_step_info", {}) or {})
        if terminated and step_info.get("bfcl_force_zero_reward"):
            reward = 0.0
        else:
            reward = self.evaluate(params={"sparse": True}) if terminated else 0.0
        # print('state_msg.simple_dict',state_msg.simple_dict)
        return {
            "state": [state_msg.simple_dict],
            "reward": reward,
            "is_terminated": terminated,
            "info": step_info,
        }

    def transition(
        self, assistant_entry: Dict[str, Any], params: Dict[str, Any]
    ) -> StateMessage:
        """执行一次 assistant 行为并让 EnvHandler 给出回应"""
        # change by czy0712
        # action_msg: ActionMessage -> assistant_entry: Dict[str, Any]

        # 记录 assistant 消息
        # assistant_entry: Dict[str, Any] = {
        #     "role": "assistant",
        #     "content": action_msg.content or "",
        # }
        # if action_msg.tool_calls:
        #     assistant_entry["tool_calls"] = [
        #         {
        #             "id": tc.id or f"call_{tc.index}",
        #             "type": "function",
        #             "function": {"name": tc.name, "arguments": tc.arguments},
        #         }
        #         for tc in action_msg.tool_calls
        #     ]

        # content to toolcalls
        assistant_entry = parse_assistant_content_to_tool_calls(
            assistant_entry,
            strict=bool(self.params.get("strict_tool_parser", True)),
            parser_mode=str(self.params.get("tool_call_parser", "xml_json")),
            reject_tool_call_with_content=bool(
                self.params.get("reject_tool_call_with_content", False)
            ),
            flag_python_literals=bool(
                self.params.get("flag_python_literals_in_tool_json", False)
            ),
            accept_parameters_as_arguments=bool(
                self.params.get("accept_parameters_as_arguments", False)
            ),
        )
        parse_error = assistant_entry.pop("_bfcl_parse_error", None)

        self.conversation_history.append(assistant_entry) ### change by czy0712

        if parse_error:
            assistant_entry["_bfcl_rejected_tool_calls"] = assistant_entry.get(
                "tool_calls", []
            )
            assistant_entry["tool_calls"] = []
            self._last_step_info = {
                **dict(getattr(self, "_last_step_info", {}) or {}),
                "bfcl_error_type": "parse_error",
                "static_fission_retryable": True,
                "static_fission_reason": parse_error,
            }
            if bool(self.params.get("terminate_on_parse_error", False)):
                self._last_step_info = {
                    **dict(getattr(self, "_last_step_info", {}) or {}),
                    "bfcl_terminated_on_parse_error": True,
                    "bfcl_force_zero_reward": True,
                }
                return StateMessage(
                    role="user",
                    content="[CONVERSATION_COMPLETED]",
                )
            err_text = f"[ERROR] Invalid tool call format: {parse_error}"
            err_text = self._enrich_parse_error(err_text)
            return StateMessage(
                role="user",
                content=err_text,
            )

        # ➜ 必须有已初始化的 handler
        if self.env_handler is None or self.original_test_entry is None:
            raise RuntimeError(
                "EnvHandler not initialised – call get_init_state() first."
            )
        # 与环境交互后env_response有两种返回情况: 
        # 1. 触发query, 附带着available tools列表, {"messages": [{"role": "user", "content": user_query}], "tools": tools} 
        # 2. 返回工具调用结果, {"messages": [{"role": "tool", "content": {<execution_results>}, 'tool_call_id': 'chatcmpl-tool-xxx'}]}
        #    <execution_results>: 正确执行时返回结果dict, e.g., {"travel_cost_list": [1140.0]}, 错误时返回error信息, e.g., {"error": "cd: temporary: No such directory. You cannot use path to change directory."}
        env_resp = self.env_handler.interact(
            self.conversation_history,
            self.original_test_entry,
            enforce_available_tools=bool(
                self.params.get("enforce_available_tools", True)
            ),
        )
        # print('env_resp in bfcl_env.py', env_resp)
        if env_resp.get("invalid_tool_call"):
            assistant_entry["_bfcl_rejected_tool_calls"] = assistant_entry.get(
                "tool_calls", []
            )
            assistant_entry["tool_calls"] = []

        return self._append_env_response(env_resp)

    def _append_env_response(
        self,
        env_resp: Dict[str, Any],
    ) -> StateMessage:
        """Append EnvHandler response to history and build the next state."""
        new_tool_calls: list[ToolCall] = []
        next_msg_content = ""
        # Track whether the last batch of messages was a tool-execution result.
        # When True and result_mode is toolace_official_prompt, we use "ipython"
        # role so that ToolACE/Llama-3.1 models see the correct token header.
        last_batch_is_tool_result = False

        result_mode = str(self.params.get("tool_result_mode", "bfcl_tool_response"))

        # FIXME: yunpeng - messages 可能是多个吗
        for idx, msg in enumerate(env_resp.get("messages", [])):
            self.conversation_history.append(msg)
            if msg["role"] == "tool":
                # FIXME 改成一次性传入所有tool messages
                rendered = tool_message_to_qwen_text(
                    msg,
                    result_mode=result_mode,
                )
                error_text = self._extract_tool_error_text(msg.get("content"))
                if error_text:
                    self._last_step_info = {
                        **dict(getattr(self, "_last_step_info", {}) or {}),
                        "bfcl_error_type": "tool_error",
                        "static_fission_retryable": True,
                        "static_fission_reason": error_text,
                    }
                rendered = self._enrich_tool_response_text(rendered, msg)
                next_msg_content += rendered
                last_batch_is_tool_result = True
            elif msg["role"] == "user":
                next_msg_content = msg.get("content", "")
                # FIXME: yunpeng 更新新的tool schema到user msg里
                self.current_turn += 1
                last_batch_is_tool_result = False
            elif msg["role"] == "env":
                # two situations:
                # 1. [{"role": "env", "content": "[CONVERSATION_COMPLETED]"}]
                # 2. [{"role": "env", "content": f"[ERROR] {error_message}"}]
                next_msg_content = msg.get("content", "")
                last_batch_is_tool_result = False

        # Both Llama-3.1 and ToolACE/watt models expect tool execution results
        # under the "ipython" role (<|start_header_id|>ipython<|end_header_id|>).
        # The official gorilla LlamaHandler explicitly uses role="ipython" for
        # all models that go through LlamaHandler (llama31_official and
        # toolace_official_prompt).  Using role="user" makes the model treat the
        # tool result as a new user query, causing severe performance degradation.
        _IPYTHON_RESULT_MODES = {"llama31_official", "toolace_official_prompt"}
        use_ipython_role = (
            last_batch_is_tool_result
            and result_mode in _IPYTHON_RESULT_MODES
        )
        state_role = Role.IPYTHON if use_ipython_role else Role.USER
        return (
            StateMessage(role=state_role, content=next_msg_content)
             ###### changed by czy0718, without tool_role, trajectory.steps cannot be converted by tool_execution_results
        )

    def _train_only_enabled(self, cfg: Dict[str, Any]) -> bool:
        if not bool(cfg.get("enable", False)):
            return False
        mode = str(self.params.get("rollout_mode", "") or "").lower()
        if mode in {"validate", "validation", "val", "test"} and not bool(
            cfg.get("apply_to_validation", False)
        ):
            return False
        categories = cfg.get("categories", None)
        if isinstance(categories, str):
            categories = [categories]
        if categories:
            category = self._test_category()
            if category not in set(str(item) for item in categories):
                return False
        return True

    def _observation_lite_config(self) -> Dict[str, Any]:
        cfg = self.params.get("observation_lite")
        return cfg if isinstance(cfg, dict) else {}

    def _observation_lite_enabled(self) -> bool:
        return self._train_only_enabled(self._observation_lite_config())

    def _observation_lite_instruction(self) -> str:
        cfg = self._observation_lite_config()
        return str(
            cfg.get(
                "instruction",
                "Observation note: parameters marked [required] must be "
                "provided exactly when calling a tool; do not invent tools "
                "outside the current schema.",
            )
        ).strip()

    def _apply_observation_lite(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not self._observation_lite_enabled():
            return tools

        cfg = self._observation_lite_config()
        add_required = bool(cfg.get("add_required_hint", True))
        add_type = bool(cfg.get("add_type_hint", False))
        add_enum = bool(cfg.get("add_enum_hint", True))
        hide_optional = bool(cfg.get("hide_optional", False))

        patched_tools = deepcopy(tools)
        for tool in patched_tools:
            func = tool.get("function", tool)
            params = func.get("parameters", {}) or {}
            props = params.get("properties", {}) or {}
            required = set(params.get("required", []) or [])

            if hide_optional:
                for key in list(props.keys()):
                    if key not in required:
                        props.pop(key, None)

            for name, spec in props.items():
                if not isinstance(spec, dict):
                    continue
                desc = str(spec.get("description", "") or "")
                additions: list[str] = []
                if add_required and name in required and "[required]" not in desc:
                    additions.append("[required]")
                if add_type and spec.get("type") and "[type:" not in desc:
                    additions.append(f"[type: {spec.get('type')}]")
                enum_values = spec.get("enum")
                if add_enum and enum_values and "Allowed values:" not in desc:
                    additions.append(
                        "Allowed values: " + ", ".join(str(v) for v in enum_values)
                    )
                if additions:
                    spec["description"] = f"{desc} {' '.join(additions)}".strip()

        return patched_tools

    # ------------------------------------------------------------------ #
    # Tool Feedback Evolution (Slot 2)
    # ------------------------------------------------------------------ #
    # When the model emits an unparseable tool_call or the tool execution
    # returns a structured error, the env replaces the terse default
    # feedback with a richer diagnostic message. Train-only by default;
    # gated by config block ``tool_feedback_evolution`` with the same
    # ``apply_to_validation`` / ``categories`` semantics as observation_lite.
    #
    # The enrichment lives on the *reactive* env interface surface: it
    # only fires when the model has already failed, so it cannot make
    # successful trajectories worse. The same dense BFCL grader pipeline
    # that A-Patch consumes still tags every turn unchanged.
    def _tool_feedback_config(self) -> Dict[str, Any]:
        cfg = self.params.get("tool_feedback_evolution")
        return cfg if isinstance(cfg, dict) else {}

    def _tool_feedback_enabled(self) -> bool:
        return self._train_only_enabled(self._tool_feedback_config())

    def _enrich_parse_error(self, base_text: str) -> str:
        if not self._tool_feedback_enabled():
            return base_text
        cfg = self._tool_feedback_config()
        max_tools = int(cfg.get("max_tool_names", 12) or 12)
        names = list(getattr(self, "_cached_tool_names", []) or [])
        if max_tools > 0 and len(names) > max_tools:
            shown = names[:max_tools] + [f"... (+{len(names) - max_tools} more)"]
        else:
            shown = names
        hints: list[str] = [base_text]
        parser_mode = str(self.params.get("tool_call_parser", ""))
        if parser_mode == "llama31_official_fc":
            hints.append(
                "[Hint] In Llama-3.1 mode, emit raw JSON only, e.g. "
                '{"name": "tool_name", "parameters": {"arg": "value"}}. '
                "Do not use <tool_call> tags."
            )
        elif parser_mode in {"toolace_fc", "toolace_official_prompt"}:
            hints.append(
                "[Hint] In ToolACE BFCL prompt mode, output only a Python "
                "function-call list, e.g. [tool_name(arg=\"value\")]. "
                "Do not use <tool_call> tags or JSON objects."
            )
        elif parser_mode == "envtuning_fc":
            hints.append(
                "[Hint] In EnvTuning BFCL mode, output exactly one <think> block "
                "followed by either one <tool_call> JSON block or one <answer> block. "
                "Tool calls must use {\"name\": \"tool_name\", \"arguments\": {...}}."
            )
        else:
            hints.append(
                "[Hint] Each tool call MUST be a JSON object inside <tool_call>...</tool_call> "
                'tags with both "name" and "arguments" keys, e.g. '
                '<tool_call>{"name": "<tool>", "arguments": {"<arg>": <value>}}</tool_call>.'
            )
        if shown:
            hints.append(
                "[Hint] Available tools: " + ", ".join(shown) + "."
            )
        hints.append(
            "[Hint] If no available tool can serve the user's request, reply in plain text "
            "instead of producing a malformed tool_call."
        )
        return "\n".join(hints)

    def _enrich_tool_response_text(
        self, rendered: str, raw_msg: Dict[str, Any]
    ) -> str:
        """Append a per-error diagnostic hint when the tool returned an error."""
        if not self._tool_feedback_enabled():
            return rendered
        content = raw_msg.get("content")
        error_text = self._extract_tool_error_text(content)
        if not error_text:
            return rendered
        cfg = self._tool_feedback_config()
        tool_name = str(raw_msg.get("name") or raw_msg.get("tool_call_id") or "")
        hint = self._diagnose_tool_error(tool_name, error_text, cfg)
        if not hint:
            return rendered
        return rendered + f"[ToolHint] {hint}\n"

    @staticmethod
    def _extract_tool_error_text(content: Any) -> str:
        if content is None:
            return ""
        if isinstance(content, dict):
            err = content.get("error")
            if isinstance(err, str):
                return err
            return ""
        if isinstance(content, str):
            text = content.strip()
            if not text:
                return ""
            if text.startswith("{") and '"error"' in text:
                try:
                    parsed = json.loads(text)
                    if isinstance(parsed, dict) and isinstance(parsed.get("error"), str):
                        return parsed["error"]
                except Exception:
                    return ""
            return ""
        return ""

    def _diagnose_tool_error(
        self, tool_name: str, error_text: str, cfg: Dict[str, Any]
    ) -> str:
        """Pattern-match common tool errors and produce a 1-line fix hint.

        The map intentionally targets the BFCL multi_turn failure modes that
        A-Patch already weights highly (state/instance/missing-arg). Patterns
        are conservative: when nothing matches the function returns "" and
        the env behaves identically to the baseline.
        """
        text = (error_text or "").strip()
        if not text:
            return ""
        lo = text.lower()
        # 1) Argument errors (TypeError / missing-required-arg style)
        m = re.search(
            r"missing\s+\d+\s+required\s+(?:positional|keyword)\s+argument(?:s)?:\s*(.+)",
            lo,
        )
        if m:
            args = m.group(1).strip().rstrip(".")
            return (
                f"Required argument(s) {args} for `{tool_name}` were missing. "
                "Re-emit the tool_call with these keys present in `arguments`."
            )
        m = re.search(r"unexpected\s+keyword\s+argument\s*['\"]?([\w_]+)", lo)
        if m:
            arg = m.group(1)
            return (
                f"`{tool_name}` does not accept argument `{arg}`. "
                "Drop it and only use arguments listed in the tool schema."
            )
        # 2) Path / instance / state errors
        if "no such directory" in lo or "no such file" in lo:
            return (
                f"The path you passed to `{tool_name}` does not exist in the current state. "
                "Inspect the latest state with a read/list-style tool first, then retry."
            )
        if "you cannot use path" in lo or "use a single component" in lo:
            return (
                f"`{tool_name}` rejects multi-component paths. "
                "Issue one navigation step at a time."
            )
        if "not authenticated" in lo or "login" in lo and "required" in lo:
            return (
                "An authentication step is required before this call can succeed. "
                "Authenticate first, then retry."
            )
        if "tool not found" in lo or "no such tool" in lo or "unknown function" in lo:
            names = list(getattr(self, "_cached_tool_names", []) or [])
            shown = ", ".join(names[:8]) if names else ""
            tail = f" Available tools start with: {shown}." if shown else ""
            return (
                f"`{tool_name}` is not in the available tool list."
                f"{tail} If no listed tool matches the user's request, reply in plain text."
            )
        # 3) Type errors (catch-all)
        if "expected" in lo and "got" in lo:
            return (
                f"`{tool_name}` rejected an argument type. "
                "Re-check argument types in the tool schema (string vs int vs list)."
            )
        # 4) Generic short error: pass through as a one-line nudge.
        if len(text) < 200:
            return (
                f"`{tool_name}` returned an error. "
                "Re-read the tool's parameter schema and retry with corrected arguments."
            )
        return ""

    # ------------------------------------------------------------------ #
    # Diagnostic-Driven Schema Evolution (Slot 3)
    # ------------------------------------------------------------------ #
    # The env reads ``capability_state.json`` written by the trainer's
    # TOCFCapabilityState. For every failure tag whose recent prevalence is
    # high AND whose recent reward_mean is low, the env appends a one-line
    # behavioral guideline to the system prompt. The guideline set therefore
    # *evolves online* as training progresses: empty in epoch 0, growing
    # with the agent's dominant failure modes, shrinking again as the
    # agent masters them.
    #
    # This is "interface evolution driven by the same diagnostic signal
    # that A-Patch consumes for advantage scaling" — i.e. the
    # co-evolution loop made concrete.
    _DEFAULT_GUIDELINES: Dict[str, str] = {
        "spurious_tool_call": (
            "[Cautious-call] If no available function clearly matches the user's request, "
            "respond in plain text rather than fabricating a tool call."
        ),
        "empty_turn_model_response": (
            "[Active-respond] When the user's request CAN be served by an available "
            "function, you MUST emit a tool_call rather than only replying in plain text."
        ),
        "state_mismatch": (
            "[State-aware] Before issuing a state-mutating call, re-read the most "
            "recent tool results to confirm the current state matches what the "
            "arguments assume."
        ),
        "instance_mismatch": (
            "[Scope] Verify that the instance/target your call refers to actually "
            "appeared in earlier tool results before calling on it."
        ),
        "response_mismatch": (
            "[Format] Match your final reply to the format implied by the user's "
            "request: lists, units, exact strings, ordering."
        ),
    }
    _STATE_CACHE_TTL_SECONDS: float = 30.0

    def _diagnostic_evolution_config(self) -> Dict[str, Any]:
        cfg = self.params.get("diagnostic_evolution")
        return cfg if isinstance(cfg, dict) else {}

    def _diagnostic_evolution_enabled(self) -> bool:
        return self._train_only_enabled(self._diagnostic_evolution_config())

    @classmethod
    def _read_capability_state(cls, path: str) -> Dict[str, Any] | None:
        if not path:
            return None
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            return None
        cache = getattr(cls, "_capability_state_cache", None)
        import time as _time
        now = _time.time()
        if (
            isinstance(cache, dict)
            and cache.get("path") == path
            and cache.get("mtime") == mtime
            and (now - cache.get("read_at", 0.0)) < cls._STATE_CACHE_TTL_SECONDS
        ):
            return cache.get("data")
        try:
            with open(path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, json.JSONDecodeError):
            return None
        cls._capability_state_cache = {
            "path": path,
            "mtime": mtime,
            "read_at": now,
            "data": data,
        }
        return data

    def _select_active_guidelines(self) -> List[str]:
        cfg = self._diagnostic_evolution_config()
        state_path = str(cfg.get("state_path") or "").strip()
        if not state_path:
            return []
        data = self._read_capability_state(state_path)
        if not data:
            return []
        # Source preference: window_tags (recent), fallback to total_tags.
        window_tags = data.get("window_tags") or {}
        total_tags = data.get("total_tags") or {}
        source = window_tags if window_tags else total_tags
        if not isinstance(source, dict) or not source:
            return []

        prev_thresh = float(cfg.get("prevalence_threshold", 0.10) or 0.10)
        reward_thresh = float(cfg.get("reward_threshold", 0.6) or 0.6)
        min_count = int(cfg.get("min_tag_count", 8) or 8)
        max_lines = int(cfg.get("max_lines", 3) or 3)
        category_scope = cfg.get("category_tags")  # optional category filter

        # Optionally narrow source via category-specific tags.
        if category_scope:
            cat_source = data.get("window_category_tags") or data.get("total_category_tags") or {}
            if isinstance(cat_source, dict):
                target_cat = self._test_category()
                merged: Dict[str, Dict[str, Any]] = {}
                for key, stats in cat_source.items():
                    if "::" not in key:
                        continue
                    cat, tag = key.split("::", 1)
                    if cat != target_cat:
                        continue
                    merged[tag] = stats
                if merged:
                    source = merged

        excluded = {"checker_error", "gt_error", "pass", "correct_abstention", "unknown"}
        total_count = sum(
            int((s or {}).get("count", 0) or 0)
            for tag, s in source.items()
            if tag not in excluded
        )
        if total_count <= 0:
            return []

        guidelines = dict(self._DEFAULT_GUIDELINES)
        custom = cfg.get("guidelines")
        if isinstance(custom, dict):
            for k, v in custom.items():
                if isinstance(v, str) and v.strip():
                    guidelines[str(k)] = v.strip()
                elif v is None:
                    guidelines.pop(str(k), None)

        candidates: list[tuple[float, str, str]] = []
        for tag, stats in source.items():
            if tag in excluded or tag not in guidelines:
                continue
            count = int((stats or {}).get("count", 0) or 0)
            if count < min_count:
                continue
            prevalence = float(count) / float(total_count)
            reward_mean = float((stats or {}).get("reward_mean", 0.0) or 0.0)
            if prevalence < prev_thresh:
                continue
            if reward_mean >= reward_thresh:
                # Tag prevalent but already largely solved → no need to nag.
                continue
            pressure = prevalence * (1.0 - reward_mean)
            candidates.append((pressure, tag, guidelines[tag]))

        candidates.sort(key=lambda x: x[0], reverse=True)
        selected = [g for _, _, g in candidates[: max(0, max_lines)]]
        return selected

    def _diagnostic_evolution_system_block(self) -> str:
        if not self._diagnostic_evolution_enabled():
            return ""
        guidelines = self._select_active_guidelines()
        if not guidelines:
            return ""
        header = "# Behavioral guidelines (env-evolved from recent failures):"
        body = "\n".join(f"- {line}" for line in guidelines)
        return f"{header}\n{body}"

    def _apply_diagnostic_evolution(
        self, tools: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        # Reserved for a future per-tool hook; today the diagnostic-driven
        # annotation lives at system-prompt scope (see
        # ``_diagnostic_evolution_system_block``). Returning the tools
        # unchanged here keeps the call-site clean and the future per-tool
        # extension a one-line swap.
        return tools

    def _test_category(self) -> str:
        entry = self.test_entry or self.original_test_entry or {}
        test_id = str(entry.get("id") or self.task_id or "")
        return test_id.rsplit("_", 1)[0] if "_" in test_id else test_id

    def evaluate(
        self,
        messages: Dict[str, Any] | None = None,
        params: Dict[str, Any] | None = None,
    ):
        """调用 EnvHandler 评估对话"""
        if self.env_handler is None:
            raise RuntimeError("EnvHandler not initialised – cannot evaluate.")

        conv_result = {
            "test_id": self.test_entry.get("id", "unknown"),
            "messages": self.conversation_history,
            "turn_count": self.current_turn,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "completed": self._is_terminated(self.conversation_history[-1]["content"]),  ### changed by czy0721
            "original_test_entry": self.original_test_entry,
        }
        sparse = (params or {}).get("sparse", False)
        result = self.env_handler.evaluate(conv_result)
        return result.get("accuracy", 0.0) if sparse else result

    def get_info(
        self,
        messages: Dict[str, Any] | None = None,
        params: Dict[str, Any] | None = None,
    ) -> str:
        return self.tools_info

    def close(self):  # Ray actor cleanup hook
        self.conversation_history.clear()

    # ------------------------------------------------------------------ #
    # 内部工具
    # ------------------------------------------------------------------ #
    def _is_terminated(self, env_content) -> bool:
        # return self.original_test_entry is not None and self.current_turn >= len(
        #     self.original_test_entry.get("question", [])
        # ) 
        # changed by czy0721, since current_turn is 0-based index
        return env_content == "[CONVERSATION_COMPLETED]"

    @staticmethod
    def _load_test_case(data_path: str, test_id: str | None) -> Dict[str, Any]:
        """按 ID / 行号加载单条 JSONL 测试用例。找不到就抛错。"""
        if not Path(data_path).exists():
            raise FileNotFoundError(f"BFCL data file '{data_path}' not found")

        if test_id is None:
            raise ValueError("task_id is required")

        with open(data_path, "r", encoding="utf-8") as f:
            if str(test_id).isdigit():
                idx = int(test_id)
                for line_no, line in enumerate(f):
                    if line_no == idx:
                        return json.loads(line)
                raise ValueError(f"Test case index {idx} not found in {data_path}")
            else:
                for line in f:
                    data = json.loads(line)
                    if data.get("id") == test_id:
                        return data
                raise ValueError(f"Test case id '{test_id}' not found in {data_path}")

    # 静态接口给 env_service 用
    @staticmethod
    def get_query_list(split: str = "train", params={"category": ["multi_turn_base"]}):
        """
        Get query list from preprocessed dataset.
        
        Args:
            split: Dataset split, either 'train' or 'test'
            params: Parameters to filter dataset (currently supports 'category')
        
        Returns:
            List of query id
        """

        path = os.getenv("BFCL_SPLID_ID_PATH")
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)[split]
            # return [json.loads(l)["id"] for l in f]
