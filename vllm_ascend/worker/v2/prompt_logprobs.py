# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from collections.abc import Callable
from functools import partial

import torch
import torch.distributed as dist
from vllm.v1.worker.gpu.sample.prompt_logprob import PromptLogprobsWorker

from vllm_ascend.distributed.parallel_state import get_lmhead_tp_group
from vllm_ascend.utils import lmhead_tp_pad_rows


class LmheadPromptLogprobsWorker(PromptLogprobsWorker):
    """Keep prompt scoring on a common LM-head collective sequence."""

    def compute_prompt_logprobs(self, logits_fn, hidden_states, *args, **kwargs):
        result = super().compute_prompt_logprobs(
            partial(self._compute_logits, logits_fn), hidden_states, *args, **kwargs
        )
        # Older vLLM versions have no fixed-token scoring stage.
        if not hasattr(PromptLogprobsWorker, "compute_prompt_token_id_logprobs"):
            self.finish_prompt_scoring(logits_fn, hidden_states)
        return result

    def compute_prompt_token_id_logprobs(self, logits_fn, hidden_states, *args, **kwargs):
        result = super().compute_prompt_token_id_logprobs(
            partial(self._compute_logits, logits_fn), hidden_states, *args, **kwargs
        )
        self.finish_prompt_scoring(logits_fn, hidden_states)
        return result

    @staticmethod
    def _agreed_rows(local_rows: int) -> int:
        rows = torch.tensor(local_rows, device="cpu", dtype=torch.int32)
        dist.all_reduce(rows, op=dist.ReduceOp.MAX, group=get_lmhead_tp_group().cpu_group)
        return int(rows.item())

    def _compute_logits(
        self, logits_fn: Callable[[torch.Tensor], torch.Tensor], hidden_states: torch.Tensor
    ) -> torch.Tensor:
        local_rows = hidden_states.shape[0]
        # Empty local chunks must not signal completion while more chunks remain.
        if local_rows == 0:
            raise ValueError("Expected a nonempty prompt scoring chunk.")
        capacity = self._agreed_rows(local_rows)
        padded = lmhead_tp_pad_rows(hidden_states, capacity, "maximum prompt scoring rows")
        return logits_fn(padded)[:local_rows]

    def finish_prompt_scoring(
        self, logits_fn: Callable[[torch.Tensor], torch.Tensor], hidden_states: torch.Tensor
    ) -> None:
        # A rank can finish either scoring path before its peers. It must still
        # join each remaining LM-head call until every rank reports zero rows.
        while capacity := self._agreed_rows(0):
            logits_fn(hidden_states.new_zeros((capacity, hidden_states.shape[-1])))
