# pixie-qa-demo

Demo for [pixie-qa](https://github.com/yiouli/pixie-qa) — automated quality assurance for AI applications.

This repository demonstrates the **improved manual instrumentation API**, which adds:

- **`observation()`** — a context manager (replacing `pixie.instrumentation.log()`) that gracefully does nothing when tracing has not been set up, so you never need to guard instrumentation code with `if tracing_active:` checks.
- **`@observe(name=...)`** — a decorator that automatically captures function inputs and outputs using [`jsonpickle`](https://github.com/jsonpickle/jsonpickle), with no boilerplate required.

Both work out of the box whether or not a tracer has been initialised, making it easy to add instrumentation incrementally and run the same code in both traced and non-traced environments.

---

## Quick Start

```python
from observe import observation, observe

# Context manager — explicit input/output control
with observation("What is the capital of France?", name="qa") as obs:
    answer = process(obs.input)
    obs.set_output(answer)
    obs.set_metadata("model", "gpt-4o")

# Decorator — inputs and outputs captured automatically via jsonpickle
@observe(name="translate")
def translate(text: str, target_lang: str) -> str:
    return call_translation_api(text, target_lang)

# Works with async functions too
@observe()
async def fetch_answer(question: str) -> str:
    return await async_llm_call(question)
```

No exception is raised if `pixie.instrumentation.init()` has not been called — the instrumentation simply becomes a no-op.

---

## Setup

```bash
pip install -r requirements.txt
# or
pip install "pixie @ git+https://github.com/yiouli/pixie-qa.git" jsonpickle
```

---

## Examples

| Script | Description |
|--------|-------------|
| [`examples/basic_observation.py`](examples/basic_observation.py) | `observation()` context manager with and without storage |
| [`examples/observe_decorator.py`](examples/observe_decorator.py) | `@observe` decorator on sync and async functions |
| [`examples/graceful_fallback.py`](examples/graceful_fallback.py) | No-exception behaviour when tracing is not set up |

Run without tracing (no errors, functions work normally):

```bash
python examples/graceful_fallback.py
```

Run with tracing enabled (spans printed to stdout):

```bash
python examples/graceful_fallback.py --init
```

Run with SQLite storage:

```bash
PIXIE_DB_PATH=demo.db python examples/basic_observation.py --store
```

---

## Running Tests

```bash
pytest tests/ -v
```

---

## API Reference

### `observation(input=None, *, name=None)`

A context manager that wraps a code block in an observation span.

- **`input`** — the input value to associate with the span (any type).
- **`name`** — an optional label for the span.
- Yields a span context with `set_output(value)` and `set_metadata(key, value)` methods.
- **Graceful fallback**: if `pixie.instrumentation.init()` has not been called, yields a no-op context object — no exception is raised.

```python
with observation("user question", name="qa") as obs:
    answer = call_llm(obs.input)
    obs.set_output(answer)
```

### `@observe(name=None)`

A decorator that automatically captures function inputs and outputs.

- **`name`** — optional span name; defaults to the function's `__name__`.
- Arguments are serialised with `jsonpickle` and stored as the span input.
- The return value is serialised and stored as the span output.
- Supports both synchronous and `async` functions.
- **Graceful fallback**: if tracing is not set up, the function runs unchanged.

```python
@observe(name="summarise")
def summarise(document: str, max_words: int = 50) -> str:
    ...

@observe()
async def fetch_data(url: str) -> dict:
    ...
```
