from __future__ import annotations

# Close system proxy
import urllib.request
urllib.request.getproxies = lambda: {}

import logging

from flask import Flask, jsonify, request

from config import settings
from prompts.registry import PROMPT_REGISTRY, resolve_prompt_name
from services.llm_service import LLMService
from services.qdrant_service import QdrantService
from utils.document_loader import extract_text_from_file, index_directory_to_qdrant, index_text_to_qdrant
from utils.logger import logger


llm_service = LLMService()
qdrant_service = QdrantService()



def build_context_from_hits(hits):
    context_parts = []
    sources = []

    for hit in hits:
        payload = hit.get("payload") or {}
        text = payload.get("text") or payload.get("content") or ""
        score = float(hit.get("score", 0) or 0)

        if not text:
            continue

        context_parts.append(text)
        sources.append(
            {
                "id": hit.get("id"),
                "score": round(score, 4),
                "source": payload.get("source"),
            }
        )
    return "\n\n".join(context_parts), sources


def query_rag(question: str, prompt_name: str | None = None):
    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    query_vector = llm_service.get_embeddings([question])[0]
    hits = qdrant_service.search(query_vector=query_vector, limit=5, score_threshold=settings.RETRIEVAL_SCORE_THRESHOLD)
    logger.info("Retrieved %d hits for the question: %s", len(hits), question[:200])
    context, sources = build_context_from_hits(hits)
    has_context = bool(context)
    chosen_prompt = resolve_prompt_name(prompt_name, has_context=has_context)
    logger.info("Using prompt '%s' for the query.", chosen_prompt)
    logger.debug("Has context: %s", has_context)
    if not has_context:
        logger.warning("No relevant documents found for the question: %s", question[:200])
        answer = llm_service.generate_answer(
            question,
            prompt_name=chosen_prompt,
        )
        return answer, [], "No relevant document found."

    logger.debug("Context length: %d characters", len(context))
    answer = llm_service.generate_answer(
        question,
        context=context,
        prompt_name=chosen_prompt,
    )
    return answer, sources, context


def create_app() -> Flask:
    app = Flask(__name__)
    app.config.from_object(settings)
    app.logger.handlers = logger.handlers
    app.logger.setLevel(logging.DEBUG if settings.DEBUG else logging.INFO)

    @app.before_request
    def log_request_start():
        if settings.DEBUG:
            app.logger.debug("Incoming request: %s %s", request.method, request.path)

    @app.get("/")
    def index():
        return jsonify({
            "message": "RAG API is running",
            "status": "ok",
        })

    @app.get("/health")
    def health_check():
        return jsonify({"status": "healthy"})

    @app.get("/api/prompts")
    def list_available_prompts():
        payload = [
            {
                "name": name,
                "module": spec.module,
                "description": spec.description,
                "required_fields": list(spec.required_fields),
                "template_path": spec.template_path,
            }
            for name, spec in sorted(PROMPT_REGISTRY.items())
        ]
        return jsonify({"prompts": payload})

    @app.post("/api/index")
    def index_documents():
        payload = request.get_json(silent=True) or {}
        text = payload.get("text") or payload.get("content") or ""
        if not text:
            return jsonify({"error": "Text content is required."}), 400

        source_name = payload.get("source", "uploaded_document")
        try:
            chunk_count = index_text_to_qdrant(
                text=text,
                source_name=source_name,
                llm_service=llm_service,
                qdrant_service=qdrant_service,
            )
            return jsonify({
                "message": "Documents indexed successfully.",
                "chunks": chunk_count,
            })
        except Exception as exc:
            return jsonify({"error": str(exc)}), 500

    @app.post("/api/upload")
    def upload_document():
        uploaded_files = request.files.getlist("file")
        if not uploaded_files:
            return jsonify({"error": "No file was uploaded."}), 400

        indexed_files = []
        total_chunks = 0

        try:
            for uploaded_file in uploaded_files:
                logger.info("Processing uploaded file: %s", uploaded_file.filename)
                if uploaded_file.filename == "":
                    continue

                text = extract_text_from_file(uploaded_file, uploaded_file.filename)
                logger.debug("Extracted text from file '%s': %d characters", uploaded_file.filename, len(text))
                if not text.strip():
                    continue

                chunk_count = index_text_to_qdrant(
                    text=text,
                    source_name=uploaded_file.filename,
                    llm_service=llm_service,
                    qdrant_service=qdrant_service,
                )
                logger.info("Indexed %d chunks from file '%s'", chunk_count, uploaded_file.filename)
                indexed_files.append(uploaded_file.filename)
                total_chunks += chunk_count

            if not indexed_files:
                return jsonify({"error": "No readable text content was found in the uploaded files."}), 400

            return jsonify({
                "message": "Files uploaded and indexed successfully.",
                "files": indexed_files,
                "chunks": total_chunks,
            })
        except Exception as exc:
            return jsonify({"error": str(exc)}), 500

    @app.post("/api/upload-folder")
    def upload_folder():
        payload = request.get_json(silent=True) or {}
        folder_path = payload.get("folder_path") or payload.get("path") or ""
        if not folder_path:
            return jsonify({"error": "Folder path is required."}), 400

        try:
            result = index_directory_to_qdrant(folder_path, llm_service=llm_service, qdrant_service=qdrant_service)
            return jsonify({
                "message": "Folder indexed successfully.",
                "files": result["files"],
                "chunks": result["chunks"],
            })
        except Exception as exc:
            return jsonify({"error": str(exc)}), 500

    @app.get("/api/documents")
    def list_documents():
        source = request.args.get("source")
        if source is not None:
            source = source.strip() or None
        try:
            return jsonify({"documents": qdrant_service.list_documents(source=source)})
        except Exception as exc:
            return jsonify({"error": str(exc)}), 500

    @app.post("/api/documents/delete")
    def delete_documents():
        payload = request.get_json(silent=True) or {}
        source = payload.get("source") or ""
        if not source:
            return jsonify({"error": "Source name is required."}), 400

        try:
            deleted_count = qdrant_service.delete_documents_by_source(source)
            return jsonify({
                "message": "Documents deleted successfully.",
                "source": source,
                "deleted_count": deleted_count,
            })
        except Exception as exc:
            return jsonify({"error": str(exc)}), 500

    @app.post("/api/clear")
    def clear_documents():
        try:
            qdrant_service.clear_collection()
            logger.info("Knowledge base cleared successfully.")
            return jsonify({"message": "Knowledge base cleared successfully."})
        except Exception as exc:
            return jsonify({"error": str(exc)}), 500

    @app.post("/api/query")
    def query_documents():
        payload = request.get_json(silent=True) or {}
        question = payload.get("question") or payload.get("query") or ""
        prompt_name = payload.get("prompt_name") or payload.get("prompt")
        logger.info("Query request received: prompt=%s, question=%s", prompt_name, question[:200])

        if not question:
            logger.warning("Empty question sent to /api/query")
            return jsonify({"error": "Question is required."}), 400

        try:
            answer, sources, context = query_rag(question, prompt_name=prompt_name)
            logger.info("Query succeeded: prompt=%s, source_count=%s", prompt_name, len(sources))
            return jsonify({
                "answer": answer,
                "sources": sources,
                "context": context,
                "prompt_name": prompt_name,
            })
        except Exception as exc:
            logger.exception("Query failed: prompt=%s, question=%s", prompt_name, question[:200])
            return jsonify({"error": str(exc)}), 500

    @app.post("/api/chat/action")
    def chat_action():
        payload = request.get_json(silent=True) or {}
        action = (payload.get("action") or "").strip().lower()
        message_id = payload.get("message_id")

        if not action:
            return jsonify({"error": "Action is required."}), 400

        supported_actions = {"like", "dislike"}
        if action not in supported_actions:
            return jsonify({"error": f"Unsupported action: {action}"}), 400

        return jsonify({
            "success": True,
            "action": action,
            "message_id": message_id,
            "message": "Action recorded successfully.",
        })

    return app


app = create_app()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=settings.DEBUG)
