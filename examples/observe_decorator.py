"""Example: using the @observe decorator.

Shows how to instrument functions automatically using the ``@observe``
decorator, which captures inputs and outputs via ``jsonpickle`` without
requiring any manual ``set_output()`` calls.

Run without tracing (no-op, no errors):

    python examples/observe_decorator.py

Run with storage:

    PIXIE_DB_PATH=demo.db python examples/observe_decorator.py --store
"""

from __future__ import annotations

import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from observe import observe


# ---------------------------------------------------------------------------
# Decorated functions
# ---------------------------------------------------------------------------


@observe(name="translate")
def translate(text: str, target_lang: str) -> str:
    """Mock translation function — inputs and outputs captured automatically."""
    translations = {
        ("hello", "fr"): "bonjour",
        ("hello", "es"): "hola",
        ("world", "fr"): "monde",
    }
    return translations.get((text.lower(), target_lang.lower()), f"[{text} in {target_lang}]")


@observe()  # name defaults to function name "summarise"
def summarise(document: str, max_words: int = 50) -> dict:
    """Mock summarisation — returns a dict to show jsonpickle handles complex types."""
    words = document.split()
    summary = " ".join(words[:max_words]) + ("..." if len(words) > max_words else "")
    return {"summary": summary, "original_word_count": len(words)}


# ---------------------------------------------------------------------------
# Async example
# ---------------------------------------------------------------------------


@observe(name="async_qa")
async def async_answer(question: str) -> str:
    """Async mock QA — observe() works with async functions too."""
    import asyncio

    await asyncio.sleep(0)  # simulate async work
    return f"Async answer to: {question}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="@observe decorator demo")
    parser.add_argument("--store", action="store_true", help="Enable SQLite storage")
    args = parser.parse_args()

    if args.store:
        from pixie import enable_storage

        enable_storage()
        print("Storage enabled — traces will be saved to SQLite.\n")
    else:
        print("Running without tracing setup — @observe will be a no-op.\n")

    # Sync decorated functions
    print("--- translate() ---")
    print(translate("hello", "fr"))
    print(translate("hello", "es"))

    print("\n--- summarise() ---")
    doc = "The quick brown fox jumps over the lazy dog. " * 5
    result = summarise(doc, max_words=10)
    print(result)

    # Async decorated function
    print("\n--- async_answer() ---")
    import asyncio

    answer = asyncio.run(async_answer("What is the meaning of life?"))
    print(answer)

    if args.store:
        import pixie.instrumentation as px

        px.flush()
        print("\nTraces flushed to storage.")


if __name__ == "__main__":
    main()
