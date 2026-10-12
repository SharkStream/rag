from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class PromptSpec:
    name: str
    template_path: str
    required_fields: tuple[str, ...]
    description: str = ""
    module: str = "general"


def _load_registry() -> dict[str, PromptSpec]:
    config_path = Path(__file__).resolve().parent / "prompt_config.yaml"
    if not config_path.exists():
        return {}

    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    registry: dict[str, PromptSpec] = {}
    for item in config.get("prompt_specs", []):
        name = item["name"]
        registry[name] = PromptSpec(
            name=name,
            template_path=item["path"],
            required_fields=tuple(item.get("required_fields", ())),
            description=item.get("description", ""),
            module=item.get("module", "general"),
        )
    return registry


PROMPT_REGISTRY: dict[str, PromptSpec] = _load_registry()

DEFAULT_PROMPT_BY_CONTEXT = {
    True: "rag.answer_with_context",
    False: "rag.answer_without_context",
}


def get_prompt_spec(name: str) -> PromptSpec:
    try:
        return PROMPT_REGISTRY[name]
    except KeyError as exc:
        raise KeyError(f"Unknown prompt: {name}") from exc


def list_prompt_names() -> tuple[str, ...]:
    return tuple(PROMPT_REGISTRY.keys())


def resolve_prompt_name(requested_name: str | None, *, has_context: bool) -> str:
    if requested_name and requested_name in PROMPT_REGISTRY:
        return requested_name

    default_name = DEFAULT_PROMPT_BY_CONTEXT.get(has_context, "rag.answer_without_context")
    if default_name in PROMPT_REGISTRY:
        return default_name

    if not PROMPT_REGISTRY:
        raise ValueError("No prompts are registered.")

    return next(iter(PROMPT_REGISTRY))
