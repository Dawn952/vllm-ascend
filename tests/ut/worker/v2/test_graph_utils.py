# SPDX-License-Identifier: Apache-2.0

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from vllm_ascend.worker.v2 import utils


@pytest.fixture
def graph_runtime(monkeypatch):
    events = []
    config = SimpleNamespace(enable_super_kernel=True)
    monkeypatch.setattr(
        utils,
        "get_ascend_config",
        lambda: SimpleNamespace(ascend_compilation_config=config),
    )
    monkeypatch.setattr(utils, "get_graph_params", lambda: "target")
    monkeypatch.setattr(utils, "get_draft_graph_params", lambda: "draft")
    monkeypatch.setattr(utils, "weak_ref_workspaces", lambda params: events.append(f"cleanup:{params}"))
    monkeypatch.setattr(utils.torch.npu, "super_kernel_scope_begin", lambda scope: events.append(f"begin:{scope}"))
    monkeypatch.setattr(utils.torch.npu, "super_kernel_scope_end", lambda scope: events.append(f"end:{scope}"))
    failure = SimpleNamespace(stage=None)
    graph_calls = []

    @contextmanager
    def capture(graph, *args, **kwargs):
        graph_calls.append((graph, args, kwargs))
        events.append("capture_begin")
        if failure.stage == "capture_begin":
            raise RuntimeError("capture_begin failed")
        try:
            yield
        finally:
            events.append("capture_end")
            if failure.stage == "capture_end":
                raise RuntimeError("capture_end failed")

    monkeypatch.setattr(utils.torch.npu, "graph", capture)

    def make_graph():
        def optimize(**kwargs):
            events.append("optimize")
            if failure.stage == "optimize":
                raise RuntimeError("optimize failed")

        return SimpleNamespace(super_kernel_optimize=Mock(side_effect=optimize))

    return SimpleNamespace(events=events, config=config, failure=failure, calls=graph_calls, make_graph=make_graph)


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("keyword_graph", [False, True])
def test_graph_capture_applies_super_kernel_before_releasing_workspaces(graph_runtime, enabled, keyword_graph):
    runtime = graph_runtime
    runtime.config.enable_super_kernel = enabled
    pool, stream = object(), object()
    # Target and draft managers each provide an independent graph object.
    for _ in range(2):
        graph = runtime.make_graph()
        if keyword_graph:
            capture = utils.torch_npu_graph_wrapper(npu_graph=graph, pool=pool, stream=stream)
        else:
            capture = utils.torch_npu_graph_wrapper(graph, pool, stream=stream)
        with capture:
            runtime.events.append("forward")

        expected = ["capture_begin"]
        if enabled:
            expected.append("begin:full_model")
        expected.append("forward")
        if enabled:
            expected.append("end:full_model")
        expected.append("capture_end")
        if enabled:
            expected.append("optimize")
        expected.extend(["cleanup:target", "cleanup:draft"])
        assert runtime.events == expected
        if enabled:
            graph.super_kernel_optimize.assert_called_once_with(optimize_options={"dcci_after_kernel_end": [".*"]})
        else:
            graph.super_kernel_optimize.assert_not_called()
        assert runtime.calls[-1] == (
            (graph, (), {"pool": pool, "stream": stream}) if keyword_graph else (graph, (pool,), {"stream": stream})
        )
        runtime.events.clear()


@pytest.mark.parametrize("stage", ["capture_begin", "forward", "capture_end", "optimize"])
def test_graph_capture_propagates_errors_and_releases_workspaces(graph_runtime, stage):
    runtime = graph_runtime
    runtime.failure.stage = stage
    graph = runtime.make_graph()
    with pytest.raises(RuntimeError, match=f"{stage} failed"), utils.torch_npu_graph_wrapper(graph):
        if stage == "forward":
            raise RuntimeError("forward failed")

    assert runtime.events[-2:] == ["cleanup:target", "cleanup:draft"]
    if stage == "optimize":
        graph.super_kernel_optimize.assert_called_once()
    else:
        graph.super_kernel_optimize.assert_not_called()
    if stage != "capture_begin":
        assert runtime.events.index("end:full_model") < runtime.events.index("capture_end")
