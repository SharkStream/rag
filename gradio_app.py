from __future__ import annotations

import gradio as gr

from config import settings
from services.llm_service import LLMService
from services.qdrant_service import QdrantService
from utils.document_loader import extract_text_from_file, index_text_to_qdrant


llm_service = LLMService()
qdrant_service = QdrantService()


def answer_question(question: str):
    """Search the vector DB for relevant context and generate an answer."""
    if not question or not question.strip():
        return "Please enter a question.", {}

    if not settings.OPENAI_API_KEY:
        return "Please configure OPENAI_API_KEY first.", {}

    try:
        query_vector = llm_service.get_embeddings([question])[0]
        hits = qdrant_service.search(query_vector=query_vector, limit=5)

        context_parts = []
        sources = []
        for hit in hits:
            payload = hit.get("payload") or {}
            text = payload.get("text") or payload.get("content") or ""
            if text:
                context_parts.append(text)
                sources.append({
                    "id": hit.get("id"),
                    "score": round(float(hit.get("score", 0)), 4),
                })

        context = "\n\n".join(context_parts)
        if not context:
            answer = llm_service.generate_answer(question)
            return answer, {"context": "No relevant document was found.", "sources": []}

        answer = llm_service.generate_answer(question, context=context)
        return answer, {"context": context, "sources": sources}
    except Exception as exc:  # pragma: no cover - UI error handling
        return f"Query failed: {exc}", {}


def index_uploaded_files(file_obj):
    """Read uploaded files, extract text, and index them into Qdrant."""
    if file_obj is None:
        return "Please upload a file first.", "", []

    if not settings.OPENAI_API_KEY:
        return "Please configure OPENAI_API_KEY first.", "", []

    files = file_obj if isinstance(file_obj, list) else [file_obj]
    indexed_count = 0
    file_names = []
    preview_text = []

    try:
        for item in files:
            if item is None:
                continue
            name = getattr(item, "name", "uploaded_file")
            text = extract_text_from_file(item, name)
            if not text.strip():
                continue
            chunk_count = index_text_to_qdrant(
                text=text,
                source_name=name,
                llm_service=llm_service,
                qdrant_service=qdrant_service,
            )
            indexed_count += chunk_count
            file_names.append(name)
            preview_text.append(text[:200])

        if not file_names:
            return "The uploaded file(s) do not contain readable text.", "", []

        return f"Indexed {indexed_count} chunks across {len(file_names)} file(s).", "\n\n".join(preview_text), file_names
    except Exception as exc:
        return f"Upload failed: {exc}", "", []


def clear_knowledge_base():
    """Clear all indexed documents from the current Qdrant collection."""
    try:
        qdrant_service.clear_collection()
        return "Knowledge base cleared successfully."
    except Exception as exc:
        return f"Clear failed: {exc}"


def delete_source_documents(source_name: str):
    """Delete all document chunks associated with the named source."""
    if not source_name or not source_name.strip():
        return "Please enter a source name to delete."

    try:
        deleted_count = qdrant_service.delete_documents_by_source(source_name.strip())
        return f"Deleted {deleted_count} chunk(s) for source: {source_name}"
    except Exception as exc:
        return f"Delete failed: {exc}"


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="RAG Chat Demo") as demo:
        gr.Markdown("# RAG Chat Demo")

        with gr.Row():
            uploaded_file = gr.Files(label="Upload one or more documents", file_types=[".txt", ".md", ".pdf", ".csv", ".json"])
            upload_btn = gr.Button("Upload and Index", variant="primary")
            clear_btn = gr.Button("Clear Knowledge Base", variant="secondary")

        upload_status = gr.Textbox(label="Upload status")
        preview = gr.Textbox(label="Preview", lines=6)
        indexed_files = gr.JSON(label="Indexed files")

        with gr.Row():
            source_name = gr.Textbox(label="Source name to delete", placeholder="e.g. doc1.txt")
            delete_btn = gr.Button("Delete Source", variant="stop")

        delete_status = gr.Textbox(label="Delete status")

        with gr.Row():
            question = gr.Textbox(
                label="Question",
                placeholder="Ask a question based on your indexed documents",
                scale=4,
            )
            submit_btn = gr.Button("Submit", variant="primary", scale=1)

        response = gr.Textbox(label="Answer", lines=8)
        context_box = gr.JSON(label="Retrieved context")

        upload_btn.click(fn=index_uploaded_files, inputs=uploaded_file, outputs=[upload_status, preview, indexed_files])
        clear_btn.click(fn=clear_knowledge_base, outputs=upload_status)
        delete_btn.click(fn=delete_source_documents, inputs=source_name, outputs=delete_status)
        submit_btn.click(fn=answer_question, inputs=question, outputs=[response, context_box])
        question.submit(fn=answer_question, inputs=question, outputs=[response, context_box])

    return demo


if __name__ == "__main__":
    demo = build_demo()
    demo.launch(server_name="0.0.0.0", server_port=7860)
