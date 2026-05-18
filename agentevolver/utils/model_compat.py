import json
import os
import tempfile
from typing import Any


LLAMA3_CHAT_TEMPLATE = (
    "{% set loop_messages = messages %}"
    "{% for message in loop_messages %}"
    "{% set content = '<|start_header_id|>' + message['role'] + "
    "'<|end_header_id|>\\n\\n' + message['content'] | trim + '<|eot_id|>' %}"
    "{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}"
    "{{ content }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}"
    "{% endif %}"
)


def _local_config_model_type(model_dir: str) -> str:
    config_path = os.path.join(model_dir, "config.json")
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return str(json.load(f).get("model_type", "") or "").lower()
    except Exception:
        return ""


def ensure_chat_template_for_local_model(local_path: str | os.PathLike[str]) -> bool:
    """Materialize a fallback chat template for Llama-family local snapshots.

    Some tool-tuned Llama checkpoints are served through vLLM's chat endpoint but
    omit `chat_template` from tokenizer_config.json. vLLM then returns a plain
    500 response for every /v1/chat/completions request. We patch only the local
    snapshot and only when the model declares itself as Llama and no template is
    already present.
    """

    model_dir = str(local_path)
    if not os.path.isdir(model_dir):
        return False
    if _local_config_model_type(model_dir) != "llama":
        return False

    tokenizer_config_path = os.path.join(model_dir, "tokenizer_config.json")
    if not os.path.exists(tokenizer_config_path):
        return False

    with open(tokenizer_config_path, "r", encoding="utf-8") as f:
        tokenizer_config: dict[str, Any] = json.load(f)

    if tokenizer_config.get("chat_template"):
        return False

    tokenizer_config["chat_template"] = LLAMA3_CHAT_TEMPLATE
    fd, tmp_path = tempfile.mkstemp(
        dir=model_dir,
        prefix=".tokenizer_config.",
        suffix=".json",
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(tokenizer_config, f, ensure_ascii=False, indent=2)
            f.write("\n")
        os.replace(tmp_path, tokenizer_config_path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise

    return True
