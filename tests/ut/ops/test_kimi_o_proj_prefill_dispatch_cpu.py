# SPDX-License-Identifier: Apache-2.0
"""Exercise the real MRv1/MRv2 prefill dispatch without importing an NPU runtime."""

import ast
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]


def _load_class(relative_path, name, scope):
    path = ROOT / relative_path
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name)
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(path), "exec"), scope)
    return scope[name]


@pytest.mark.parametrize("v2", [False, True])
@pytest.mark.parametrize("pure_prefill", [None, False, True])
@pytest.mark.parametrize("fallback", [None, "empty", "fp32", "weight_stride", "comm_limit"])
def test_prefill_dispatch_reads_runner_specific_context(v2, pure_prefill, fallback):
    context = SimpleNamespace(additional_kwargs={})
    if pure_prefill is not None:
        if v2:
            context.additional_kwargs["is_pure_prefill"] = pure_prefill
        else:
            context.is_pure_prefill = pure_prefill
    proxy_scope = dict(
        Any=Any,
        get_forward_context=lambda: context,
        envs_vllm=SimpleNamespace(VLLM_USE_V2_MODEL_RUNNER=v2),
    )
    proxy = _load_class("vllm_ascend/ascend_forward_context.py", "_ExtraForwardContextProxy", proxy_scope)()
    calls = []
    rank, world = 3, 8

    def reduce_scatter(partial):
        padded = torch.nn.functional.pad(partial, (0, 0, 0, (-partial.shape[0]) % world))
        return (padded * world).chunk(world, dim=0)[rank]

    def fused(x, weight, hcom, size, **kwargs):
        calls.append("fused")
        assert hcom == "tp8" and size == world
        assert kwargs == {"reduce_op": "sum", "comm_mode": "ai_cpu"}
        return reduce_scatter(x @ weight), torch.empty(0)

    def fallback_rs(partial):
        calls.append("fallback")
        return reduce_scatter(partial)

    scope = dict(
        torch=torch,
        CustomRowParallelOp=object,
        UnquantizedLinearMethod=object,
        is_forward_context_available=lambda: True,
        _EXTRA_CTX=proxy,
        torch_npu=SimpleNamespace(npu_quant_mm_reduce_scatter=fused),
        sp_reduce_scatter=fallback_rs,
    )
    cls = _load_class("vllm_ascend/ops/linear_op.py", "KimiOProjMMReduceScatterOp", scope)
    op = cls.__new__(cls)
    op.get_input_parallel = lambda x: x
    op.world_size, op.hcom = world, "tp8"
    dtype = torch.float32 if fallback == "fp32" else torch.bfloat16
    tokens = 0 if fallback == "empty" else 9
    inputs = torch.ones(tokens, 256, dtype=dtype)
    weight = torch.full((32, 256), 0.125, dtype=dtype)
    if fallback == "weight_stride":
        weight = torch.stack((weight, weight), dim=-1)[..., 0]
    if fallback == "comm_limit":
        op.MAX_COMM_BYTES = 16 * 32 * weight.element_size()
    op.layer = SimpleNamespace(weight=weight)
    op.quant_method = SimpleNamespace(apply=lambda layer, x, bias: torch.nn.functional.linear(x, layer.weight))
    output, bias = op.apply_impl(inputs)
    expected = reduce_scatter(torch.nn.functional.linear(inputs, weight))
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    assert bias is None
    assert calls == ["fused" if pure_prefill is True and fallback is None else "fallback"]


def test_kimi_o_projection_fusion_remains_enabled_by_default():
    tree = ast.parse((ROOT / "vllm_ascend/ascend_config.py").read_text(encoding="utf-8"))
    fields = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id == "enable_kimi_o_proj_mm_reduce_scatter"
    ]
    assert len(fields) == 1
    assert ast.literal_eval(fields[0].value) is True
