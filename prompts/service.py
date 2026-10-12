from __future__ import annotations

from typing import Any

from prompts.render import PromptContext, render_prompt
from prompts.registry import get_prompt_spec, list_prompt_names


class PromptService:
    def render(self, name: str, **kwargs: Any) -> str:
        return render_prompt(name, **kwargs)

    def render_context(self, name: str, ctx: PromptContext | dict[str, Any] | None = None, **kwargs: Any) -> str:
        if isinstance(ctx, PromptContext):
            merged = {**ctx.as_dict(), **kwargs}
        elif isinstance(ctx, dict):
            merged = {**ctx, **kwargs}
        else:
            merged = dict(kwargs)
        return render_prompt(name, **merged)

    def get_spec(self, name: str):
        return get_prompt_spec(name)

    def list_prompts(self) -> tuple[str, ...]:
        return list_prompt_names()
