"""Improved manual instrumentation API for pixie-qa.

This module provides convenience wrappers on top of ``pixie.instrumentation``
that improve usability:

- :func:`observation` — context manager renamed from ``log()``, with graceful
  no-op behaviour when tracing has not been set up.
- :func:`observe` — decorator that automatically captures function inputs and
  outputs as JSON using ``jsonpickle``.

Both work without requiring :func:`pixie.instrumentation.init` to have been
called first; if tracing is not set up they silently do nothing.

Example usage::

    from observe import observation, observe

    # Context manager — explicit input/output control
    with observation("What is the capital of France?", name="qa") as obs:
        answer = ask_llm(obs.input)
        obs.set_output(answer)

    # Decorator — inputs and outputs captured automatically
    @observe(name="translate")
    def translate(text: str, target_lang: str) -> str:
        return call_translation_api(text, target_lang)
"""

from __future__ import annotations

import functools
import logging
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any, Callable

import jsonpickle

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# No-op span context — used when tracing is not initialised
# ---------------------------------------------------------------------------


class _NoopSpanContext:
    """A do-nothing stand-in for ``_SpanContext`` when tracing is disabled."""

    input: Any = None

    def __init__(self, input: Any = None) -> None:
        self.input = input

    def set_output(self, output_value: Any) -> None:
        """No-op: tracing is not set up."""

    def set_metadata(self, key: str, value: Any) -> None:
        """No-op: tracing is not set up."""


# ---------------------------------------------------------------------------
# observation() context manager
# ---------------------------------------------------------------------------


def _is_tracing_active() -> bool:
    """Return True if pixie instrumentation has been initialised."""
    try:
        import pixie.instrumentation as px

        return bool(px._state.initialized)  # type: ignore[attr-defined]
    except Exception:
        return False


@contextmanager
def observation(
    input: Any = None,  # noqa: A002
    *,
    name: str | None = None,
) -> Generator[Any, None, None]:
    """Context manager that wraps a code block in an observation span.

    Unlike ``pixie.instrumentation.log()``, this function does **not** raise
    an exception when tracing has not been set up; it yields a no-op context
    object instead so that application code can be written once and work both
    with and without a live tracer.

    Args:
        input: The input value to associate with the span.
        name:  An optional label for the observation span.

    Yields:
        A span context with ``set_output()`` and ``set_metadata()`` methods.
        If tracing is not active, yields a :class:`_NoopSpanContext` that
        silently ignores all calls.

    Example::

        with observation("user question", name="qa") as obs:
            answer = process(obs.input)
            obs.set_output(answer)
    """
    if not _is_tracing_active():
        yield _NoopSpanContext(input=input)
        return

    import pixie.instrumentation as px

    with px.log(input=input, name=name) as span:
        yield span


# ---------------------------------------------------------------------------
# @observe decorator
# ---------------------------------------------------------------------------


def observe(name: str | None = None) -> Callable:
    """Decorator that automatically captures function inputs and outputs.

    Wraps the decorated function in an :func:`observation` span.  Arguments
    are serialised with ``jsonpickle`` and stored as the span input; the
    return value is serialised and stored as the span output.

    Args:
        name: Optional span name. Defaults to the decorated function's
              ``__name__``.

    Returns:
        A decorator that wraps the target function.

    Example::

        @observe(name="translate")
        def translate(text: str, target_lang: str) -> str:
            return call_translation_api(text, target_lang)

        # Works with async functions too
        @observe()
        async def fetch_answer(question: str) -> str:
            return await async_llm_call(question)
    """

    def decorator(func: Callable) -> Callable:
        span_name = name if name is not None else func.__name__

        if _is_async(func):

            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                input_data = _serialize_inputs(func, args, kwargs)
                with observation(input=input_data, name=span_name) as obs:
                    result = await func(*args, **kwargs)
                    obs.set_output(_serialize_output(result))
                return result

            return async_wrapper

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            input_data = _serialize_inputs(func, args, kwargs)
            with observation(input=input_data, name=span_name) as obs:
                result = func(*args, **kwargs)
                obs.set_output(_serialize_output(result))
            return result

        return sync_wrapper

    return decorator


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _is_async(func: Callable) -> bool:
    import asyncio

    return asyncio.iscoroutinefunction(func)


def _serialize_inputs(func: Callable, args: tuple, kwargs: dict) -> str:
    """Serialise positional and keyword arguments as a JSON-compatible string."""
    import inspect

    try:
        sig = inspect.signature(func)
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        return str(jsonpickle.encode(dict(bound.arguments)))
    except Exception as exc:
        logger.debug("observe: could not serialise inputs: %s", exc)
        return repr({"args": args, "kwargs": kwargs})


def _serialize_output(value: Any) -> str:
    """Serialise a return value as a JSON-compatible string."""
    try:
        return str(jsonpickle.encode(value))
    except Exception as exc:
        logger.debug("observe: could not serialise output: %s", exc)
        return repr(value)
