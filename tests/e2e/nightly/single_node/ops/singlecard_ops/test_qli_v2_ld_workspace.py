# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

import pytest
import torch
import torch_npu

import vllm_ascend
from vllm_ascend.utils import bootstrap_custom_op_env

bootstrap_custom_op_env(include_vendor_lib=True)
import vllm_ascend.vllm_ascend_C  # type: ignore[import-untyped] # noqa: E402,F401


@pytest.mark.parametrize("rows", [6, 8, 16])
@torch.inference_mode()
def test_qli_v2_ld_later_s1_block(rows):
    """Check split-S2 reduction beyond the first S1 block against a CPU oracle."""
    soc = torch_npu.npu.get_soc_version()
    if not (220 <= soc <= 225 or 250 <= soc <= 256):
        pytest.skip("requires the A2/A3 arch22 QLI kernel")
    torch.set_num_threads(8)
    print("QLI_PACKAGE", vllm_ascend.__file__, flush=True)
    ops = torch.ops._C_ascend
    generator = torch.Generator().manual_seed(20261109 + rows)
    heads, width, page, length, topk = 32, 128, 128, 32768, 2048
    blocks = length // page
    query_cpu = torch.randint(-48, 48, (rows, heads, width), generator=generator, dtype=torch.int8)
    logical_key = torch.randint(-48, 48, (blocks, page, 1, width), generator=generator, dtype=torch.int8)
    weights_cpu = (torch.rand((rows, heads), generator=generator) + 0.25).half()
    query_scale_cpu = (torch.rand((rows, heads), generator=generator) * 0.02 + 0.01).half()
    logical_key_scale = (torch.rand((blocks, page, 1), generator=generator) + 0.5).half()
    pages = torch.randperm(blocks, generator=generator)
    key_cpu = torch.empty_like(logical_key)
    key_scale_cpu = torch.empty_like(logical_key_scale)
    key_cpu[pages] = logical_key
    key_scale_cpu[pages] = logical_key_scale

    query = query_cpu.npu()
    common = dict(
        cu_seqlens_q=torch.tensor([0, rows], dtype=torch.int32, device="npu"),
        seqused_k=torch.tensor([length], dtype=torch.int32, device="npu"),
        max_seqlen_q=rows,
        layout_q="TND",
        layout_k="PA_BBND",
        mask_mode=3,
        cmp_ratio=1,
        # arch22 enables LD through this switch; world=1 retains ordinary causal semantics.
        enable_dcp=True,
        cp_world=1,
        cp_rank=0,
        cp_kv_cache_interleave_size=page,
        causal_seqlens_kv_global=torch.tensor([length], dtype=torch.int32, device="npu"),
    )
    metadata = ops.npu_quant_lightning_indexer_v2_metadata(
        heads, 1, width, topk, 2, batch_size=1, max_seqlen_k=length, device=str(query.device), **common
    )
    # QLI v2 metadata ABI: 36 AIC entries, followed by 72 LD entries of 8 int32 fields.
    aic_entries, ld_entries, entry_width = 36, 72, 8
    ld = metadata.cpu().flatten()[aic_entries * entry_width : (aic_entries + ld_entries) * entry_width]
    ld = ld.reshape(ld_entries, entry_width)
    active = ld[ld[:, 0] != 0]
    assert active.numel() and ((active[:, 2] >= 1) & (active[:, 4] > 1)).any(), (
        "test must exercise split-S2 LD on a later S1 block", active.tolist()
    )
    print("LD_METADATA", json.dumps({"rows": rows, "entries": active.tolist()}), flush=True)

    indices, values = ops.npu_quant_lightning_indexer_v2(
        query,
        key_cpu.npu(),
        weights_cpu.npu(),
        query_scale_cpu.npu(),
        key_scale_cpu.npu(),
        topk,
        2,
        block_table=pages.to(torch.int32).unsqueeze(0).npu(),
        metadata=metadata,
        return_value=1,
        **common,
    )
    indices_cpu = indices.cpu().reshape(rows, topk).long()
    values_cpu = values.cpu().reshape(rows, topk).float()
    dots = torch.einsum("qhd,kd->qhk", query_cpu.float(), logical_key.reshape(length, width).float())
    dots = (dots / 1024).clamp_min(0).half().float()
    scores = (dots * (weights_cpu * query_scale_cpu).float().unsqueeze(-1)).sum(1)
    scores *= logical_key_scale.reshape(length).float()
    for row in range(rows):
        visible = length - rows + row + 1
        selected = indices_cpu[row]
        assert selected.unique().numel() == topk, f"duplicate indices in row {row}"
        assert ((selected >= 0) & (selected < visible)).all(), f"invalid indices in row {row}"
        expected = scores[row, :visible].topk(topk).values
        torch.testing.assert_close(
            scores[row, selected].sort(descending=True).values,
            expected,
            rtol=2e-2,
            atol=2e-3,
            msg=lambda msg: f"wrong top-k in row {row}: {msg}",
        )
        torch.testing.assert_close(
            values_cpu[row],
            scores[row, selected],
            rtol=2e-2,
            atol=2e-3,
            msg=lambda msg: f"wrong scores in row {row}: {msg}",
        )
    print("ACCURACY_OK", json.dumps({"rows": rows, "length": length, "topk": topk}), flush=True)
