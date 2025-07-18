# SPDX-License-Identifier: Apache-2.0
"""A layer that samples the next tokens from the model's outputs."""
# ─── 额外依赖 ──────────────────────────────────────────────────────────────
import math, bisect
from collections import deque
import torch
# 全局可调参数
_ENT_UPDATE_STEP = 0  # 初始化全局计数器

if "_GLOBAL_TEMP_CFG" not in globals():
    _GLOBAL_TEMP_CFG = {
        "window_size": 100_000,  # 最近多少个 token 的熵
        "percentile" : 0.40,     # 前 p% 熵 → 高温
        "T_base"     : 1.0,      # 正常温度
        "T_max"      : 1.2,       # 高温
        "UPDATE_INTERVAL": 49
    }
# 滑动窗口：保存最近 window_size 个熵，并维护一个升序副本
_ENT_WINDOW = deque(maxlen=_GLOBAL_TEMP_CFG["window_size"])
_ENT_SORTED = []
print("Init _ENT_WINDOW || Init _ENT_SORTED ")
def set_entropy_temp_cfg(**kwargs):
    """在运行时动态调整窗口大小/温度。"""
    global _GLOBAL_TEMP_CFG, _ENT_WINDOW, _ENT_SORTED
    _GLOBAL_TEMP_CFG.update(kwargs)
    if "window_size" in kwargs:
        _ENT_WINDOW = deque(list(_ENT_WINDOW),
                            maxlen=_GLOBAL_TEMP_CFG["window_size"])
        _ENT_SORTED.clear()
        print("Init _ENT_WINDOW || Init _ENT_SORTED ")
        print('_GLOBAL_TEMP_CFG',_GLOBAL_TEMP_CFG)

import torch
import torch.nn as nn

from vllm.v1.outputs import LogprobsTensors, SamplerOutput
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.ops.bad_words import apply_bad_words
from vllm.v1.sample.ops.penalties import (apply_all_penalties,
                                          apply_min_token_penalties)
from vllm.v1.sample.ops.topk_topp_sampler import TopKTopPSampler

_SAMPLING_EPS = 1e-5


class Sampler(nn.Module):

    def __init__(self):
        super().__init__()
        self.topk_topp_sampler = TopKTopPSampler()

    def forward(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> SamplerOutput:
        # NOTE(woosuk): Use the original logits (before any penalties or
        # temperature scaling) for the top-k logprobs.
        # This is different from the V0 sampler, which uses the logits that
        # is used for sampling (after penalties and temperature scaling).
        # TODO(rob): provide option for logprobs post sampling.
        # See https://vllm-dev.slack.com/archives/C07UUL8E61Z/p1735907856007919 # noqa: E501
        num_logprobs = sampling_metadata.max_num_logprobs
        return_temp_only = (
            num_logprobs == 1             # 你设定的信号
        )

        if num_logprobs is not None:
            raw_logprobs = self.compute_logprobs(logits)

        # Use float32 for the logits.
        logits = logits.to(torch.float32)
        # Apply allowed token ids.
        logits = self.apply_allowed_token_ids(logits, sampling_metadata)
        # Apply bad words exclusion.
        logits = self.apply_bad_words(logits, sampling_metadata)
        # Apply logits bias.
        logits = self.apply_logits_bias(logits, sampling_metadata)
        # Apply penalties (e.g., min_tokens, freq_penalties).
        logits = self.apply_penalties(logits, sampling_metadata)

        # ------------------------------------------------------------------

        # Sample the next token.
        '''sampled = self.sample(logits, sampling_metadata)'''
        sampled, temp_vec = self.sample(logits, sampling_metadata)
        # Convert sampled token ids to int64 (long) type to ensure compatibility
        # with subsequent operations that may use these values as indices.
        # This conversion is necessary because FlashInfer sampling operations
        # return int32 (while PyTorch argmax and topk return int64).
        sampled = sampled.long()

        # Gather the logprobs of the topk and sampled token (if requested).
        # Get logprobs and rank tensors (if requested)
        logprobs_tensors = None if num_logprobs is None else \
            self.gather_logprobs(raw_logprobs, num_logprobs, token_ids=sampled)

        # Use int32 to reduce the tensor size.
        sampled = sampled.to(torch.int32)
        if num_logprobs == 1:
            
            # 把该列 token_id 设为 -1，logprob 改成温度
            # torch.set_printoptions(edgeitems=3,    # 每维保留几项
            #            threshold=20,   # 总元素数阈值，超过则省略
            #            linewidth=120,  # 一行最长字符
            #            precision=4,    # 小数点后位数
            #            sci_mode=False) # 不用科学计数法
            logprobs_tensors.logprob_token_ids[:, 0].fill_(-1)        # token_id = -1

            # logprobs_tensors.logprob_token_ids.fill_(-1)
            # print('logprobs_tensors.logprobs before', logprobs_tensors.logprobs)
            logprobs_tensors.logprobs[:, 0].copy_(temp_vec.to(
                logprobs_tensors.logprobs.dtype
            ))
            # print('logprobs_tensors.logprobs after', logprobs_tensors.logprobs)


        # These are GPU tensors.
        sampler_output = SamplerOutput(
            # The sampled tokens are expanded to 2D tensor with shape
            # [num_requests, 1], where each row represents one generated
            # token per request.
            sampled_token_ids=sampled.unsqueeze(-1),
            logprobs_tensors=logprobs_tensors,
        )
        return sampler_output

    def apply_temperature(
        self,
        logits: torch.Tensor,
        temp: torch.Tensor,
    ) -> torch.Tensor:
        # Use in-place division to avoid creating a new tensor.
        return logits.div_(temp.unsqueeze(dim=1))

    def greedy_sample(self, logits: torch.Tensor) -> torch.Tensor:
        return logits.argmax(dim=-1).view(-1)

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> torch.Tensor:
        """Sample logits based on sampling metadata.

        The various logits processing functions called in this method
        may update the logits tensor in-place.
        """

        assert not (sampling_metadata.all_greedy
                    and sampling_metadata.all_random)
        if sampling_metadata.all_random:
            greedy_sampled = None
        else:
            greedy_sampled = self.greedy_sample(logits)
            if sampling_metadata.all_greedy:
                return greedy_sampled

        assert sampling_metadata.temperature is not None

        # ---------- ★ 熵-分位温控逻辑插入点 ★ -------------------------------
        # print("log here")
        cfg = _GLOBAL_TEMP_CFG
        # if cfg and cfg["window_size"] > 0  and (sampling_metadata.temperature == 1.0).any().item():
        # apply adaptive temperature only for training sampling        
        
        if cfg and cfg["window_size"] > 0 and (sampling_metadata.temperature != 0.6).any().item():  
            global _ENT_WINDOW, _ENT_SORTED, _ENT_UPDATE_STEP

            # 1) 计算当前 batch 的熵
            probs = torch.softmax(logits, dim=-1, dtype=torch.float)
            ent   = -(probs * torch.log(probs + 1e-9)).sum(dim=-1)        # [B]

            # 2) 更新滑动窗口 & 升序表
            for e in ent.tolist():
                _ENT_UPDATE_STEP += 1
                if len(_ENT_WINDOW) == _ENT_WINDOW.maxlen:
                    if _ENT_UPDATE_STEP % cfg["UPDATE_INTERVAL"] == 0: 
                        _ENT_UPDATE_STEP = 0
                        continue
                    old = _ENT_WINDOW[0]
                    idx = bisect.bisect_left(_ENT_SORTED, old)
                    if idx < len(_ENT_SORTED) and _ENT_SORTED[idx] == old:
                        _ENT_SORTED.pop(idx)
                _ENT_WINDOW.append(e)
                bisect.insort(_ENT_SORTED, e)

            # 3) 计算 (1-p) 分位阈值
            if _ENT_SORTED:
                k = max(0, int((1.0 - cfg["percentile"])
                            * (len(_ENT_SORTED) - 1)))
                threshold = _ENT_SORTED[k]
            else:
                threshold = float("inf")

            # 4) 构造温度向量并缩放 logits
            T_vec = torch.full_like(ent, cfg["T_base"])
            T_vec[ent >= threshold] = cfg["T_max"] + sampling_metadata.temperature[ent >= threshold] - cfg["T_base"]
            final_temp = T_vec
            logits = logits / T_vec.unsqueeze(1)
        else:
            # Apply temperature.
            final_temp = sampling_metadata.temperature
            logits = self.apply_temperature(logits, sampling_metadata.temperature)
        


        # Apply min_p.
        if sampling_metadata.min_p is not None:
            logits = self.apply_min_p(logits, sampling_metadata.min_p)

        # Apply top_k and/or top_p.
        random_sampled = self.topk_topp_sampler(
            logits,
            sampling_metadata.generators,
            sampling_metadata.top_k,
            sampling_metadata.top_p,
        )

        if greedy_sampled is None:
            return random_sampled, final_temp

        sampled = torch.where(
            sampling_metadata.temperature < _SAMPLING_EPS,
            greedy_sampled,
            random_sampled,
            out=greedy_sampled,  # Reuse tensor
        )
        return sampled, final_temp

    def compute_logprobs(self, logits: torch.Tensor) -> torch.Tensor:
        return logits.log_softmax(dim=-1, dtype=torch.float32)

    def gather_logprobs(
        self,
        logprobs: torch.Tensor,
        num_logprobs: int,
        token_ids: torch.Tensor,
    ) -> LogprobsTensors:
        """
        Gather logprobs for topk and sampled/prompt token.

        Args:
          logprobs: (num tokens) x (vocab) tensor
          num_logprobs: minimum number of logprobs to
                        retain per token
          token_ids: prompt tokens (if prompt logprobs)
                     or sampled tokens (if sampled
                     logprobs); 1D token ID tensor
                     with (num tokens) elements
                     Must be int64.

        Returns:
          Top-k int indices tensor, (num tokens) x (num_logprobs + 1)
          Top-k float logprobs tensor, (num tokens) x (num_logprobs + 1)
          Sampled token rank tensor, (num tokens)
        """
        assert token_ids.dtype == torch.int64
        # Find the topK values.
        topk_logprobs, topk_indices = torch.topk(logprobs,
                                                 num_logprobs,
                                                 dim=-1)

        # Get with the logprob of the prompt or sampled token.
        token_ids = token_ids.unsqueeze(-1)
        token_logprobs = logprobs.gather(-1, token_ids)

        # Compute the ranks of the actual token.
        token_ranks = (logprobs >= token_logprobs).sum(-1)

        # Concatenate together with the topk.
        indices = torch.cat((token_ids, topk_indices), dim=1)
        logprobs = torch.cat((token_logprobs, topk_logprobs), dim=1)

        # Use int32 to reduce the tensor size.
        indices = indices.to(torch.int32)

        return LogprobsTensors(indices, logprobs, token_ranks)

    def apply_penalties(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> torch.Tensor:
        if sampling_metadata.min_tokens:
            apply_min_token_penalties(logits,
                                      sampling_metadata.output_token_ids,
                                      sampling_metadata.min_tokens)
        if not sampling_metadata.no_penalties:
            assert sampling_metadata.prompt_token_ids is not None
            logits = apply_all_penalties(
                logits,
                sampling_metadata.prompt_token_ids,
                sampling_metadata.presence_penalties,
                sampling_metadata.frequency_penalties,
                sampling_metadata.repetition_penalties,
                sampling_metadata.output_token_ids,
            )
        return logits

    def apply_min_p(
        self,
        logits: torch.Tensor,
        min_p: torch.Tensor,
    ) -> torch.Tensor:
        """
        Filters logits using adaptive probability thresholding.
        """
        # Convert logits to probability distribution
        probability_values = torch.nn.functional.softmax(logits, dim=-1)
        # Calculate maximum probabilities per sequence
        max_probabilities = torch.amax(probability_values,
                                       dim=-1,
                                       keepdim=True)
        # Reshape min_p for broadcasting
        adjusted_min_p = min_p.unsqueeze(1) * max_probabilities
        # Identify valid tokens using threshold comparison
        valid_token_mask = probability_values >= adjusted_min_p
        # Apply mask using boolean indexing
        logits[~valid_token_mask] = -float('inf')
        return logits

    def apply_logits_bias(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> torch.Tensor:
        # TODO(houseroad): this implementation is extremely inefficient.
        # One idea is implement this as a PyTorch C++ op, and we may
        # even optimize the logit_bias layout.

        # Get vocabulary size from logits
        vocab_size = logits.shape[-1]

        for i, logit_bias in enumerate(sampling_metadata.logit_bias):
            if logit_bias:
                for token_id, bias in logit_bias.items():
                    # Check token_id bounds to ensure within vocabulary
                    if token_id < 0 or token_id >= vocab_size:
                        raise ValueError(
                            f"token_id {token_id} in logit_bias contains "
                            f"out-of-vocab token id. Vocabulary size: "
                            f"{vocab_size}")
                    logits[i, token_id] += bias
        return logits

    def apply_allowed_token_ids(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> torch.Tensor:
        if sampling_metadata.allowed_token_ids_mask is not None:
            logits.masked_fill_(sampling_metadata.allowed_token_ids_mask,
                                float("-inf"))
        return logits

    def apply_bad_words(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> torch.Tensor:
        if sampling_metadata.bad_words_token_ids:
            apply_bad_words(
                logits,
                sampling_metadata.bad_words_token_ids,
                sampling_metadata.output_token_ids,
            )
        return logits
