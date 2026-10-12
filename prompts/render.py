from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, StrictUndefined, select_autoescape

from prompts.registry import get_prompt_spec


@dataclass
class PromptContext:
    question: str = ""
    context: str = ""
    language: str = "Chinese"
    system_role: str = "You are a helpful assistant."
    user_name: str | None = None
    doc_titles: list[str] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        data = {
            "question": self.question,
            "context": self.context,
            "language": self.language,
            "system_role": self.system_role,
            "user_name": self.user_name,
            "doc_titles": self.doc_titles or [],
            "metadata": self.metadata or {},
        }
        return data


def _get_env() -> Environment:
    template_dir = Path(__file__).resolve().parent / "templates"
    return Environment(
        loader=FileSystemLoader(str(template_dir)),
        autoescape=select_autoescape(enabled_extensions=("html", "xml"), default_for_string=True),
        undefined=StrictUndefined,
    )


def render_prompt(name: str, **kwargs: Any) -> str:
    spec = get_prompt_spec(name)

    rendered_values = dict(kwargs)
    if not rendered_values:
        rendered_values = {}

    for field_name in spec.required_fields:
        if field_name not in rendered_values:
            rendered_values[field_name] = ""

    template = _get_env().get_template(spec.template_path)
    return template.render(**rendered_values)
