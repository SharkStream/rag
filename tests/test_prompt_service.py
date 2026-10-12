from prompts.service import PromptService


def test_prompt_service_renders_template_variables() -> None:
    service = PromptService()
    result = service.render(
        "rag.answer_with_context",
        question="What is RAG?",
        context="RAG combines retrieval and generation.",
        language="English",
    )

    assert "What is RAG?" in result
    assert "RAG combines retrieval and generation." in result
    assert "Answer in English." in result


def test_prompt_service_supports_summary_module() -> None:
    service = PromptService()
    result = service.render(
        "summary.summarize",
        content="This is a long document summary example.",
        language="English",
    )

    assert "This is a long document summary example." in result
    assert "Summary:" in result
