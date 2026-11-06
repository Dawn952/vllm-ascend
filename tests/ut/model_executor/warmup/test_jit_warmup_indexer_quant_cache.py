# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from tests.ut.model_executor.warmup.helpers import make_mock_worker
from vllm_ascend.model_executor.warmup.kernel_warmup import kernel_warmup
from vllm_ascend.ops.triton import indexer_quant_cache_int8 as kernels


def test_bound_cache_layout_is_registered_and_compiled_without_launching():
    key = torch.empty_strided((8, 128, 1, 128), (16800, 128, 128, 1), dtype=torch.int8)
    scale = torch.empty_strided((8, 128, 1, 1), (8400, 1, 1, 1), dtype=torch.float16)
    backend = SimpleNamespace(
        _fused_k_cache=kernels.indexer_quant_cache_int8,
        _pcp_active=False,
        k_cache=SimpleNamespace(kv_cache=(key, scale)),
    )
    layer = SimpleNamespace(impl=SimpleNamespace(indexer=SimpleNamespace(impl=backend)))
    worker = make_mock_worker()
    worker.vllm_config.compilation_config.static_forward_context = {"layer0": layer, "layer1": layer}
    kernel = Mock(arg_names=kernels._indexer_quant_cache_int8_kernel.arg_names)
    with (
        patch.object(kernels._INDEXER_QUANT_CACHE_INT8_KERNEL, "kernel", kernel),
        patch("vllm_ascend.model_executor.warmup.kernel_warmup._register_dspark_swa_indices_warmup", return_value=0),
    ):
        kernel_warmup(worker)

    calls = [call.kwargs for call in kernel.warmup.call_args_list]
    assert len(calls) == 32
    assert {(call["INPUT_ROWS"], call["slots_ptr"].dtype, call["slots_ptr"].data_ptr()) for call in calls} == {
        (rows, dtype, pointer)
        for rows in (1, 2, 4, 8, 16, 32, 64, 128)
        for dtype in (torch.int32, torch.int64)
        for pointer in (0, 1)
    }
    for call in calls:
        assert call["grid"] == (1,)
        assert call["NUM_SLOTS"] == 1024
        assert call["KEY_PAGE_STRIDE"] == 16800
        assert call["SCALE_PAGE_STRIDE"] == 8400
        assert call["X_ROW_STRIDE"] == 128
        assert call["X_COL_STRIDE"] == call["SLOT_STRIDE"] == 1
        assert call["key_ptr"].data_ptr() == call["scale_ptr"].data_ptr() == call["x_ptr"].data_ptr() == 0
