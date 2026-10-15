from invoke import task


@task
def check(c):
    """Run environment preflight checks for Redis, Qdrant, and embedding config."""
    c.run("python -m utils.preflight_check")


@task
def eval(c):
    """Run the evaluation suite against the project retrieval pipeline."""
    c.run("python -m tests.eval")


@task
def app(c, host="0.0.0.0", port=5000):
    """Start the Flask app through uvicorn as an ASGI-compatible entrypoint."""
    c.run(f"uvicorn app:asgi_app --host {host} --port {port}")


@task
def serve(c, host="0.0.0.0", port=5000):
    """Alias for app startup."""
    app(c, host=host, port=port)


# if __name__ == "__main__":
#     from tests.eval import main as eval_main
#     eval_main()