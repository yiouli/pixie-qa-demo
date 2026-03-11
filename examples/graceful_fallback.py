"""Example: graceful fallback when tracing is not set up.

One of the key usability improvements is that ``observation()`` and
``@observe`` do **not** raise exceptions when tracing has not been
initialised.  This example demonstrates that applications can be written
once and run in two modes:

  1. Without tracing (e.g. during local development or in tests that
     do not care about telemetry) — functions run normally, no errors.
  2. With tracing enabled — spans are captured and delivered to handlers.

Run without tracing:

    python examples/graceful_fallback.py

Run with tracing enabled:

    python examples/graceful_fallback.py --init
"""

from __future__ import annotations

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from observe import observation, observe


@observe(name="pipeline_step")
def process_text(text: str) -> str:
    return text.upper()


def run_pipeline(items: list[str]) -> list[str]:
    results = []
    for item in items:
        with observation(item, name="pipeline") as obs:
            result = process_text(item)
            obs.set_output(result)
            results.append(result)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Graceful fallback demo")
    parser.add_argument(
        "--init",
        action="store_true",
        help="Call pixie.instrumentation.init() to enable tracing",
    )
    args = parser.parse_args()

    if args.init:
        import pixie.instrumentation as px

        class PrintHandler(px.InstrumentationHandler):
            async def on_llm(self, span: px.LLMSpan) -> None:
                pass

            async def on_observe(self, span: px.ObserveSpan) -> None:
                print(
                    f"  [SPAN] name={span.name!r}  "
                    f"input={str(span.input)[:40]!r}  "
                    f"output={str(span.output)[:40]!r}"
                )

        px.init()
        px.add_handler(PrintHandler())
        print("Tracing initialised — spans will be printed.\n")
    else:
        print(
            "Tracing NOT initialised — observation() and @observe are no-ops.\n"
            "No errors will be raised.\n"
        )

    inputs = ["hello world", "the quick brown fox", "pixie-qa demo"]
    print("Processing items:")
    results = run_pipeline(inputs)
    for original, result in zip(inputs, results):
        print(f"  {original!r} -> {result!r}")

    if args.init:
        import pixie.instrumentation as px

        px.flush()
        print("\nTracing complete.")


if __name__ == "__main__":
    main()
