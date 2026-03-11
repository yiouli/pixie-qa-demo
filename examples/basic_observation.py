"""Example: using the observation() context manager.

Shows how to instrument a function manually using the improved
``observation()`` API, including the graceful no-op behaviour when tracing
has not been set up.

Run without setting up tracing (no output, no errors):

    python examples/basic_observation.py

Run with storage enabled (traces saved to SQLite):

    PIXIE_DB_PATH=demo.db python examples/basic_observation.py --store
"""

from __future__ import annotations

import argparse
import sys
import os

# Allow running from the repo root without installing the package.
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from observe import observation


def answer_question(question: str) -> str:
    """A mock QA function instrumented with observation()."""
    with observation(question, name="qa") as obs:
        # Simulate processing
        answer = f"The answer to '{question}' is 42."
        obs.set_output(answer)
        obs.set_metadata("model", "mock-v1")
    return answer


def main() -> None:
    parser = argparse.ArgumentParser(description="Basic observation() demo")
    parser.add_argument(
        "--store",
        action="store_true",
        help="Enable SQLite storage (requires PIXIE_DB_PATH env var or uses demo.db)",
    )
    args = parser.parse_args()

    if args.store:
        from pixie import enable_storage

        enable_storage()
        print("Storage enabled — traces will be saved to SQLite.\n")
    else:
        print("Running without tracing setup — observation() will be a no-op.\n")

    questions = [
        "What is the capital of France?",
        "What is 2 + 2?",
        "Who wrote Hamlet?",
    ]

    for q in questions:
        answer = answer_question(q)
        print(f"Q: {q}")
        print(f"A: {answer}\n")

    if args.store:
        import pixie.instrumentation as px

        px.flush()
        print("Traces flushed to storage.")


if __name__ == "__main__":
    main()
