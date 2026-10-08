from __future__ import annotations

from flask import Flask, jsonify, request

from config import settings
from services.llm_service import LLMService
from services.qdrant_service import QdrantService
from utils.document_loader import extract_text_from_file, index_text_to_qdrant


llm_service = LLMService()
qdrant_service = QdrantService()


def build_context_from_hits(hits):
    context_parts = []
    sources = []
    for hit in hits:
        payload = hit.get("payload") or {}
        text = payload.get("text") or payload.get("content") or ""
        if not text:
            continue
        context_parts.append(text)
        sources.append(
            {
                "id": hit.get("id"),
                "score": round(float(hit.get("score", 0)), 4),
                "source": payload.get("source"),
            }
        )
    return "\n\n".join(context_parts), sources


def query_rag(question: str):
    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    query_vector = llm_service.get_embeddings([question])[0]
    hits = qdrant_service.search(query_vector=query_vector, limit=5)
    context, sources = build_context_from_hits(hits)

    if not context:
        answer = llm_service.generate_answer(question)
        return answer, [], "No relevant document found."

    answer = llm_service.generate_answer(question, context=context)
    return answer, sources, context


def create_app() -> Flask:
    app = Flask(__name__)
    app.config.from_object(settings)

    @app.get("/")
    def index():
        return jsonify({
            "message": "RAG API is running",
            "status": "ok",
        })

    @app.get("/health")
    def health_check():
        return jsonify({"status": "healthy"})

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
                if uploaded_file.filename == "":
                    continue

                text = extract_text_from_file(uploaded_file, uploaded_file.filename)
                if not text.strip():
                    continue

                chunk_count = index_text_to_qdrant(
                    text=text,
                    source_name=uploaded_file.filename,
                    llm_service=llm_service,
                    qdrant_service=qdrant_service,
                )
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

    @app.get("/api/documents")
    def list_documents():
        source = request.args.get("source")
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
            return jsonify({"message": "Knowledge base cleared successfully."})
        except Exception as exc:
            return jsonify({"error": str(exc)}), 500

    @app.post("/api/query")
    def query_documents():
        payload = request.get_json(silent=True) or {}
        question = payload.get("question") or payload.get("query") or ""
        if not question:
            return jsonify({"error": "Question is required."}), 400

        try:
            answer, sources, context = query_rag(question)
            return jsonify({
                "answer": answer,
                "sources": sources,
                "context": context,
            })
        except Exception as exc:
            return jsonify({"error": str(exc)}), 500

    return app


app = create_app()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=settings.DEBUG)
