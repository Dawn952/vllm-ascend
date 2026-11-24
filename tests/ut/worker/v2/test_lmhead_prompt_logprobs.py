# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from unittest.mock import Mock, patch

import torch
from vllm.v1.worker.gpu.sample.prompt_logprob import PromptLogprobsWorker

from vllm_ascend.worker.v2.prompt_logprobs import LmheadPromptLogprobsWorker


def test_prompt_scoring_preserves_local_rows_and_joins_later_chunks():
    worker = LmheadPromptLogprobsWorker(2, torch.device("cpu"))
    hidden = torch.arange(6, dtype=torch.float32).reshape(3, 2)
    scheduled_chunks = []

    def logits_fn(rows):
        scheduled_chunks.append(rows.clone())
        return rows * 2

    def score(logits_fn, rows):
        return logits_fn(rows)

    with (
        patch.object(PromptLogprobsWorker, "compute_prompt_logprobs", side_effect=score),
        patch.object(PromptLogprobsWorker, "compute_prompt_token_id_logprobs", side_effect=score),
        patch.object(worker, "_agreed_rows", side_effect=[4, 3, 2, 0]) as agreed,
    ):
        prompt = worker.compute_prompt_logprobs(logits_fn, hidden[:2])
        fixed = worker.compute_prompt_token_id_logprobs(logits_fn, hidden[2:])

    torch.testing.assert_close(prompt, hidden[:2] * 2)
    torch.testing.assert_close(fixed, hidden[2:] * 2)
    assert [call.args[0] for call in agreed.call_args_list] == [2, 1, 0, 0]
    assert [chunk.shape[0] for chunk in scheduled_chunks] == [4, 3, 2]
    assert not scheduled_chunks[0][2:].any()
    assert not scheduled_chunks[1][1:].any()
    assert not scheduled_chunks[2].any()


def test_idle_rank_joins_every_prompt_scoring_chunk_and_skips_empty_step():
    worker = LmheadPromptLogprobsWorker(2, torch.device("cpu"))
    hidden = torch.ones(1, 2)
    logits_fn = Mock()
    with patch.object(worker, "_agreed_rows", side_effect=[3, 1, 0, 0]) as agreed:
        worker.finish_prompt_scoring(logits_fn, hidden)
        worker.finish_prompt_scoring(logits_fn, hidden)

    assert [call.args[0] for call in agreed.call_args_list] == [0, 0, 0, 0]
    assert [call.args[0].shape for call in logits_fn.call_args_list] == [(3, 2), (1, 2)]
    assert all(not call.args[0].any() for call in logits_fn.call_args_list)
