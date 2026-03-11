"""Tests for observe.py — improved manual instrumentation API."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from observe import _NoopSpanContext, _serialize_inputs, _serialize_output, observation, observe


# ---------------------------------------------------------------------------
# _NoopSpanContext
# ---------------------------------------------------------------------------


def test_noop_span_context_stores_input() -> None:
    ctx = _NoopSpanContext(input="hello")
    assert ctx.input == "hello"


def test_noop_span_context_set_output_does_not_raise() -> None:
    ctx = _NoopSpanContext()
    ctx.set_output("result")  # must not raise


def test_noop_span_context_set_metadata_does_not_raise() -> None:
    ctx = _NoopSpanContext()
    ctx.set_metadata("key", "value")  # must not raise


# ---------------------------------------------------------------------------
# observation() — no tracing set up
# ---------------------------------------------------------------------------


def test_observation_yields_noop_when_tracing_not_active() -> None:
    """observation() must not raise when tracing is not initialised."""
    with observation("input", name="test") as obs:
        obs.set_output("output")
        obs.set_metadata("k", "v")
    # reaching here means no exception was raised


def test_observation_noop_context_has_correct_input() -> None:
    with observation("my input") as obs:
        assert obs.input == "my input"


def test_observation_no_input_defaults_to_none() -> None:
    with observation() as obs:
        assert obs.input is None


# ---------------------------------------------------------------------------
# observation() — with tracing active
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_px_log():
    """Patch pixie.instrumentation to simulate an active tracer."""
    mock_span = MagicMock()
    mock_span.set_output = MagicMock()
    mock_span.set_metadata = MagicMock()

    from contextlib import contextmanager

    @contextmanager
    def fake_log(input=None, *, name=None):  # noqa: A002
        mock_span.input = input
        mock_span.name = name
        yield mock_span

    with (
        patch("observe._is_tracing_active", return_value=True),
        patch("pixie.instrumentation.log", fake_log),
    ):
        yield mock_span


def test_observation_delegates_to_px_log_when_active(mock_px_log: MagicMock) -> None:
    with observation("the question", name="qa") as obs:
        obs.set_output("the answer")

    mock_px_log.set_output.assert_called_once_with("the answer")


def test_observation_passes_input_to_px_log(mock_px_log: MagicMock) -> None:
    with observation("input value", name="step") as obs:
        assert obs.input == "input value"


# ---------------------------------------------------------------------------
# @observe — sync function
# ---------------------------------------------------------------------------


def test_observe_calls_wrapped_function() -> None:
    @observe(name="test_fn")
    def add(a: int, b: int) -> int:
        return a + b

    result = add(2, 3)
    assert result == 5


def test_observe_returns_original_result_when_tracing_inactive() -> None:
    @observe()
    def greet(name: str) -> str:
        return f"Hello, {name}!"

    assert greet("World") == "Hello, World!"


def test_observe_uses_function_name_as_default_span_name() -> None:
    captured_names: list[str | None] = []

    from contextlib import contextmanager

    @contextmanager
    def fake_observation(input=None, *, name=None):  # noqa: A002
        ctx = _NoopSpanContext(input=input)
        captured_names.append(name)
        yield ctx

    with patch("observe.observation", fake_observation):

        @observe()
        def my_custom_function() -> str:
            return "result"

        my_custom_function()

    assert captured_names == ["my_custom_function"]


def test_observe_uses_provided_name() -> None:
    captured_names: list[str | None] = []

    from contextlib import contextmanager

    @contextmanager
    def fake_observation(input=None, *, name=None):  # noqa: A002
        ctx = _NoopSpanContext(input=input)
        captured_names.append(name)
        yield ctx

    with patch("observe.observation", fake_observation):

        @observe(name="custom_name")
        def some_function() -> str:
            return "result"

        some_function()

    assert captured_names == ["custom_name"]


def test_observe_sets_output_on_span() -> None:
    outputs: list[Any] = []

    class CapturingNoop(_NoopSpanContext):
        def set_output(self, value: Any) -> None:
            outputs.append(value)

    from contextlib import contextmanager

    @contextmanager
    def fake_observation(input=None, *, name=None):  # noqa: A002
        yield CapturingNoop(input=input)

    with patch("observe.observation", fake_observation):

        @observe()
        def double(x: int) -> int:
            return x * 2

        double(7)

    # The output should be the jsonpickle-encoded value
    import jsonpickle

    assert outputs == [str(jsonpickle.encode(14))]


# ---------------------------------------------------------------------------
# @observe — async function
# ---------------------------------------------------------------------------


def test_observe_async_function() -> None:
    @observe(name="async_step")
    async def async_double(x: int) -> int:
        return x * 2

    result = asyncio.run(async_double(5))
    assert result == 10


def test_observe_async_returns_original_result() -> None:
    @observe()
    async def async_greet(name: str) -> str:
        return f"Hi, {name}!"

    result = asyncio.run(async_greet("Alice"))
    assert result == "Hi, Alice!"


# ---------------------------------------------------------------------------
# Input serialisation
# ---------------------------------------------------------------------------


def test_serialize_inputs_positional() -> None:
    def fn(a: int, b: str) -> None:
        pass

    result = _serialize_inputs(fn, (1, "hello"), {})
    assert "a" in result
    assert "b" in result


def test_serialize_inputs_keyword() -> None:
    def fn(a: int, b: str = "default") -> None:
        pass

    result = _serialize_inputs(fn, (1,), {"b": "custom"})
    assert "custom" in result


def test_serialize_output_string() -> None:
    import jsonpickle

    result = _serialize_output("hello")
    assert result == str(jsonpickle.encode("hello"))


def test_serialize_output_dict() -> None:
    import jsonpickle

    data = {"key": "value", "num": 42}
    result = _serialize_output(data)
    assert result == str(jsonpickle.encode(data))
