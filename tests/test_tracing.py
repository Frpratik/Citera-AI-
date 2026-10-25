import uuid

from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from app.observability.metrics import Metrics
from app.observability.tracing import (
    Trace,
    TraceCallback,
    current_trace,
    span,
    use_trace,
)


def llm_result(input_tokens: int, output_tokens: int) -> LLMResult:
    message = AIMessage(
        content="x",
        usage_metadata={
            "input_tokens": input_tokens, "output_tokens": output_tokens, "total_tokens": input_tokens + output_tokens,
        },
    )
    return LLMResult(generations=[[ChatGeneration(message=message)]])


def test_a_span_is_recorded_in_the_current_trace_and_the_metrics():
    trace, metrics = Trace(), Metrics()
    with use_trace(trace), span("retrieve_documents", metrics):
        pass
    assert [s["name"] for s in trace.spans] == ["retrieve_documents"]
    assert 'chat_stage_seconds_count{stage="retrieve_documents"} 1.0' in metrics.render()[0].decode()


def test_spans_without_a_trace_are_harmless():
    assert current_trace.get() is None
    with span("anything"):  # nothing is recording; this must simply work
        pass


def test_the_trace_is_cleared_when_the_block_ends():
    with use_trace(Trace()):
        assert current_trace.get() is not None
    assert current_trace.get() is None


def test_model_calls_are_recorded_with_tokens_and_the_step_that_made_them():
    trace, metrics = Trace(), Metrics()
    callback, run = TraceCallback(metrics), uuid.uuid4()
    with use_trace(trace), span("plan_question"):
        callback.on_chat_model_start({}, [[]], run_id=run, invocation_params={"model": "model-a"})
        callback.on_llm_end(llm_result(120, 30), run_id=run)

    [call] = trace.llm_calls
    assert (call.model, call.stage, call.input_tokens, call.output_tokens, call.outcome) == (
        "model-a", "plan_question", 120, 30, "ok",
    )
    assert trace.input_tokens == 120 and trace.output_tokens == 30
    text = metrics.render()[0].decode()
    assert 'llm_tokens_total{kind="input",model="model-a"} 120.0' in text
    assert 'llm_calls_total{model="model-a",outcome="ok"} 1.0' in text


def test_the_model_name_given_at_creation_is_used_for_every_call():
    trace, metrics = Trace(), Metrics()
    callback, run = TraceCallback(metrics, model_name="the-real-model"), uuid.uuid4()
    with use_trace(trace):
        callback.on_chat_model_start({}, [[]], run_id=run, invocation_params={})  # no name in the call itself
        callback.on_llm_end(llm_result(10, 5), run_id=run)
    assert trace.llm_calls[0].model == "the-real-model"
    assert 'llm_calls_total{model="the-real-model",outcome="ok"} 1.0' in metrics.render()[0].decode()


def test_a_failed_model_call_is_recorded_as_an_error():
    trace = Trace()
    callback, run = TraceCallback(), uuid.uuid4()
    with use_trace(trace):
        callback.on_chat_model_start({}, [[]], run_id=run, invocation_params={"model": "model-a"})
        callback.on_llm_error(RuntimeError("down"), run_id=run)
    assert trace.llm_calls[0].outcome == "error" and trace.input_tokens == 0


def test_the_summary_adds_everything_up():
    trace, callback = Trace(), TraceCallback()
    with use_trace(trace):
        for tokens in (100, 40):
            run = uuid.uuid4()
            callback.on_chat_model_start({}, [[]], run_id=run, invocation_params={"model": "m"})
            callback.on_llm_end(llm_result(tokens, 10), run_id=run)
        trace.event("fallback:backup-model")
    summary = trace.summary()
    assert summary["llm_calls"] == 2 and summary["input_tokens"] == 140 and summary["output_tokens"] == 20
    assert summary["fallback_used"] is True and summary["events"] == ["fallback:backup-model"]
