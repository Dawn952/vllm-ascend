# SPDX-License-Identifier: Apache-2.0
"""Allocation descriptors and ownership for uncompressed paired SFA caches."""

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from vllm.v1.kv_cache_interface import KVCacheGroupSpec, MambaSpec

from vllm_ascend.core.kv_cache_interface import AscendMLAAttentionSpec, AscendSFAIndexerCacheSpec
from vllm_ascend.models.kimi_k3.paired_sfa_cache import (
    paired_sfa_cache_config,
    paired_sfa_cache_groups,
    paired_sfa_pool_bytes,
)
from vllm_ascend.patch.platform import patch_kv_cache_utils
from vllm_ascend.worker.v2 import attn_utils


def make_specs():
    specs = {}
    sparse = [*range(3, 92, 4), 92]
    for layer in range(93):
        name = f"model.layers.{layer}"
        if layer in sparse:
            specs[name + ".attn"] = AscendMLAAttentionSpec(
                block_size=1536,
                num_kv_heads=1,
                head_size=656,
                dtype=torch.int8,
                cache_sparse_sfa_c8=True,
            )
            specs[name + ".indexer.k_cache"] = AscendSFAIndexerCacheSpec(
                block_size=1536,
                num_kv_heads=1,
                head_size=128,
                dtype=torch.int8,
                scale_dim=1,
                scale_dtype=torch.float32,
                cache_sparse_li_c8=True,
            )
        else:
            specs[name + ".linear_attn"] = MambaSpec(
                block_size=1536,
                shapes=((3, 4608), (12, 128, 128)),
                dtypes=(torch.bfloat16, torch.float32),
                mamba_cache_mode="align",
            )
    return specs


def make_config():
    return SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(model_type="kimi_k31"),
            hf_text_config=SimpleNamespace(model_type="kimi_linear"),
            max_model_len=64,
        ),
        speculative_config=None,
        parallel_config=SimpleNamespace(decode_context_parallel_size=1, prefill_context_parallel_size=1),
        scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=False),
        cache_config=SimpleNamespace(num_gpu_blocks_override=None, prefix_cache_retention_interval=None),
        additional_config={},
        kv_transfer_config=None,
    )


def test_token_blocks_unchanged_and_components_share_ids():
    specs, config = make_specs(), make_config()
    groups = paired_sfa_cache_groups(config, specs)
    assert len(groups) == 4
    assert [len(g.layer_names) for g in groups] == [48, 23, 23, 23]
    assert {s.block_size for s in specs.values()} == {1536}
    assert {s.page_size_bytes for s in specs.values()} == {1007616, 202752}
    assert paired_sfa_pool_bytes(groups) == 29048832
    plan = paired_sfa_cache_config(config, groups, 17556701184)
    assert plan.num_blocks == 604
    assert {t.size for t in plan.kv_cache_tensors} == {604 * 29048832}
    assert plan.kv_cache_tensors[0].size <= 17556701184
    assert len(plan.kv_cache_tensors) == 28
    assert len(groups[0].kv_cache_spec.kv_cache_specs) == 48
    assert all(t.layer_stride == plan.num_blocks * t.block_stride for t in plan.kv_cache_tensors)


@pytest.mark.parametrize("interval", [None, 5])
def test_paired_config_propagates_prefix_cache_retention_interval(interval):
    config = make_config()
    config.cache_config.prefix_cache_retention_interval = interval
    groups = paired_sfa_cache_groups(config, make_specs())
    plan = paired_sfa_cache_config(config, groups, 17556701184)
    assert plan.prefix_cache_retention_interval == interval


def test_pipeline_projection_retains_only_local_slot_cost():
    groups = paired_sfa_cache_groups(make_config(), make_specs())
    projected = [
        KVCacheGroupSpec([n for n in g.layer_names if int(n.split(".")[2]) < 24], g.kv_cache_spec) for g in groups
    ]
    assert paired_sfa_pool_bytes(projected) == 6 * (1007616 + 202752)


def test_mismatched_component_blocks_rejected():
    specs = make_specs()
    name = next(n for n in specs if n.endswith(".k_cache"))
    specs[name] = replace(specs[name], block_size=3072)
    with pytest.raises(ValueError, match="same local token block"):
        paired_sfa_cache_groups(make_config(), specs)


@pytest.mark.parametrize("model_type", ["kimi_k31", "kimi_k3", "deepseek_v32", "glm_moe_dsa"])
def test_paired_planner_selection_uses_outer_model_type(monkeypatch, model_type):
    config, specs = make_config(), make_specs()
    config.model_config.hf_config.model_type = model_type
    paired = Mock(return_value=[])
    upstream = Mock(return_value=[])
    monkeypatch.setattr(patch_kv_cache_utils, "paired_sfa_cache_groups", paired)
    monkeypatch.setattr(patch_kv_cache_utils, "_orig_get_kv_cache_groups", upstream)
    monkeypatch.setattr(patch_kv_cache_utils, "get_qwen4_exp_kv_cache_groups", lambda *_: None)

    actual = patch_kv_cache_utils._get_ascend_kv_cache_groups(config, specs)

    # The same hybrid cache geometry must not opt unrelated models into K3.1.
    # K3.1's nested text config still reports kimi_linear.
    selected, other = (paired, upstream) if model_type == "kimi_k31" else (upstream, paired)
    assert actual is selected.return_value
    selected.assert_called_once_with(config, specs)
    other.assert_not_called()


def test_alignment_is_idempotent_and_discards_obsolete_state_padding():
    specs = {
        n: replace(s, page_size_padded=66502656) if isinstance(s, MambaSpec) else s for n, s in make_specs().items()
    }
    first = paired_sfa_cache_groups(make_config(), specs)
    second = paired_sfa_cache_groups(make_config(), specs)
    assert paired_sfa_pool_bytes(first) == paired_sfa_pool_bytes(second) == 29048832


def test_a3_int8_fp16_pages_preserve_tp16_recurrent_ownership():
    specs = {}
    for name, spec in make_specs().items():
        if isinstance(spec, MambaSpec):
            spec = replace(spec, shapes=((3, 2304), (6, 128, 128)))
        elif isinstance(spec, AscendSFAIndexerCacheSpec):
            spec = replace(spec, scale_dtype=torch.float16)
        specs[name] = replace(spec, block_size=768)
    config = make_config()
    groups = paired_sfa_cache_groups(config, specs)
    # 24 packed-MLA pages (768*656 bytes) plus 24 INT8/FP16 indexer
    # pages (768*(128+2) bytes); recurrent checkpoints alias main slots.
    assert {spec.page_size_bytes for spec in specs.values()} == {503808, 99840}
    assert {spec.block_size for spec in specs.values()} == {768}
    assert paired_sfa_pool_bytes(groups) == 14487552
    plan = paired_sfa_cache_config(config, groups, 115900539)
    assert plan.num_blocks == 8
    assert {tensor.size for tensor in plan.kv_cache_tensors} == {115900416}
    assert all(
        spec.sfa_dcp_replicated_indexer_size == 1
        for spec in specs.values()
        if isinstance(spec, AscendSFAIndexerCacheSpec)
    )


def make_small_specs(with_draft):
    specs = {}
    for layer in range(6):
        name = f"model.layers.{layer}"
        if layer in (2, 5):
            specs[name + ".attn"] = AscendMLAAttentionSpec(
                block_size=16, num_kv_heads=1, head_size=64, dtype=torch.int8, cache_sparse_sfa_c8=True
            )
            specs[name + ".indexer.k_cache"] = AscendSFAIndexerCacheSpec(
                block_size=16,
                num_kv_heads=1,
                head_size=12,
                dtype=torch.int8,
                scale_dim=1,
                scale_dtype=torch.float32,
            )
        else:
            specs[name + ".linear_attn"] = MambaSpec(
                block_size=16,
                shapes=((2, 4), (4, 4)),
                dtypes=(torch.bfloat16, torch.float32),
                mamba_cache_mode="align",
            )
    if with_draft:
        for layer, head_size in enumerate((16, 24)):
            specs[f"draft.layers.{layer}.attn"] = AscendMLAAttentionSpec(
                block_size=16,
                num_kv_heads=1,
                head_size=head_size,
                dtype=torch.bfloat16,
                is_dspark_draft=True,
                non_causal_multi_token_decode=True,
            )
    return specs


@pytest.mark.parametrize("with_draft", [False, True])
@pytest.mark.parametrize("kv_transfer", [False, True])
def test_paired_backing_preserves_layer_and_block_ownership(monkeypatch, with_draft, kv_transfer):
    config, specs = make_config(), make_small_specs(with_draft)
    config.kv_transfer_config = SimpleNamespace(kv_connector="MooncakeConnectorV2") if kv_transfer else None
    groups = paired_sfa_cache_groups(config, specs)
    budget = 15360 if with_draft else 10240
    plan = paired_sfa_cache_config(config, groups, budget)
    assert plan.num_blocks == 4
    assert {tensor.size for tensor in plan.kv_cache_tensors} == {budget}
    monkeypatch.setattr(attn_utils, "get_current_vllm_config", lambda: config)
    monkeypatch.setattr(attn_utils, "enable_sfa", lambda _: True)
    raw = attn_utils._allocate_kv_cache(plan, shared_layers={}, device=torch.device("cpu"))

    # One allocation, even though many placement descriptors refer to it.
    assert set(raw) == set(specs)
    assert len({tensor.untyped_storage().data_ptr() for tensor in raw.values()}) == 1
    alignment = 2 * 1024 * 1024 if kv_transfer else 0
    assert {tensor.untyped_storage().nbytes() for tensor in raw.values()} == {budget + alignment}
    base = raw["model.layers.2.attn"].data_ptr()
    if kv_transfer:
        assert base % alignment == 0
    expected_offsets = {
        "model.layers.2.attn": 0,
        "model.layers.5.attn": 4096,
        "model.layers.0.linear_attn": 0,
        "model.layers.3.linear_attn": 4096,
        "model.layers.1.linear_attn": 0,
        "model.layers.4.linear_attn": 4096,
        "model.layers.2.indexer.k_cache": 8192,
        "model.layers.5.indexer.k_cache": 9216,
    }
    if with_draft:
        expected_offsets.update({"draft.layers.0.attn": 10240, "draft.layers.1.attn": 12288})
        assert groups[0].is_eagle_group
        assert all(name in groups[0].layer_names for name in expected_offsets if name.startswith("draft."))
    for name, offset in expected_offsets.items():
        assert raw[name].data_ptr() == base + offset
        assert raw[name].numel() == 4 * specs[name].page_size_bytes

    # Each group owns a different live pool ID. Writing every layer's page
    # must preserve all other live layers, including indexer and draft data.
    live_pages = []
    for group_id, group in enumerate(groups):
        block_id = group_id + 1
        for name in group.layer_names:
            page = specs[name].page_size_bytes
            view = raw[name][block_id * page : (block_id + 1) * page]
            value = len(live_pages) + 1
            view.fill_(value)
            live_pages.append((view, value))
    for view, value in live_pages:
        assert torch.all(view == value)

    # KDA state reshaping must retain the nonzero layer offset and page stride.
    name = "model.layers.3.linear_attn"
    states = attn_utils._reshape_mamba_kv_cache(raw[name], specs[name])
    assert states[0].data_ptr() == base + 4096
    assert states[1].data_ptr() == base + 4096 + 16
    for state in states:
        assert state.stride(0) * state.element_size() == 1024

    backend = SimpleNamespace(get_kv_cache_shape=lambda blocks, block, heads, dim: (blocks, block, heads, dim))
    indexer_groups = [
        SimpleNamespace(kv_cache_group_id=0, kv_cache_spec=spec, layer_names=[name], backend=backend)
        for name, spec in specs.items()
        if isinstance(spec, AscendSFAIndexerCacheSpec)
    ]
    indexer_caches = attn_utils._reshape_kv_cache_v2(
        attn_groups=indexer_groups,
        kv_cache_raw_tensors=raw,
        cache_dtype="auto",
        kernel_block_sizes=[16],
        shared_kv_cache_layers={},
        kv_cache_config=plan,
    )
    for name, (key, scale) in indexer_caches.items():
        assert key.data_ptr() == base + expected_offsets[name]
        assert scale.data_ptr() == key.data_ptr() + 192
        assert key.stride(0) * key.element_size() == scale.stride(0) * scale.element_size() == 256
        assert scale.dtype == torch.float32
        assert key.shape == (4, 16, 1, 12)
        assert scale.shape == (4, 16, 1, 1)

    if with_draft:
        for layer, nope_dim in enumerate((8, 16)):
            name = f"draft.layers.{layer}.attn"
            key, rope = attn_utils._reshape_combined_mla_kv_cache(
                raw[name], (4, 16, 1, nope_dim), (4, 16, 1, 8), torch.bfloat16, specs[name].page_size_bytes
            )
            assert key.data_ptr() == base + expected_offsets[name]
            assert rope.data_ptr() == key.data_ptr() + 16 * nope_dim * 2
            assert key.stride(0) * 2 == rope.stride(0) * 2 == specs[name].page_size_bytes
            assert key.dtype == rope.dtype == torch.bfloat16


@pytest.mark.parametrize("local_role", ["target", "recurrent", "draft"])
def test_paired_backing_pipeline_local_groups(monkeypatch, local_role):
    config, specs = make_config(), make_small_specs(with_draft=True)
    groups = paired_sfa_cache_groups(config, specs)
    if local_role == "target":
        local_names = {name for name in specs if not name.startswith("draft.")}
        budget = 10240
    elif local_role == "recurrent":
        local_names = {name for name, spec in specs.items() if isinstance(spec, MambaSpec)}
        budget = 8192
    else:
        local_names = {name for name in specs if name.startswith("draft.")}
        budget = 5120
    # Upstream PP projection retains global specs while filtering local names.
    groups = [
        replace(group, layer_names=[name for name in group.layer_names if name in local_names]) for group in groups
    ]
    plan = paired_sfa_cache_config(config, groups, budget)
    assert plan.num_blocks == 4
    assert {tensor.size for tensor in plan.kv_cache_tensors} == {budget}
    monkeypatch.setattr(attn_utils, "get_current_vllm_config", lambda: config)
    monkeypatch.setattr(attn_utils, "enable_sfa", lambda _: True)
    raw = attn_utils._allocate_kv_cache(plan, shared_layers={}, device=torch.device("cpu"))
    assert set(raw) == local_names
    assert len({tensor.untyped_storage().data_ptr() for tensor in raw.values()}) == 1
    assert {tensor.untyped_storage().nbytes() for tensor in raw.values()} == {budget}
    if local_role == "draft":
        assert raw["draft.layers.1.attn"].data_ptr() - raw["draft.layers.0.attn"].data_ptr() == 2048


@pytest.mark.parametrize("with_draft", [False, True])
def test_paired_backing_keeps_zero_block_plan(with_draft):
    config, specs = make_config(), make_small_specs(with_draft)
    groups = paired_sfa_cache_groups(config, specs)
    plan = paired_sfa_cache_config(config, groups, 0)
    assert plan.num_blocks == 0
    assert all(tensor.size == tensor.offset == tensor.layer_stride == 0 for tensor in plan.kv_cache_tensors)
