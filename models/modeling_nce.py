from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class InfoNCEOutput:
    logits: torch.Tensor
    selected: torch.Tensor
    loss: Optional[torch.Tensor] = None
    hits_at_1: Optional[torch.Tensor] = None
    easy_loss: Optional[torch.Tensor] = None


def masked_log_softmax(vector: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    ``torch.nn.functional.log_softmax(vector)`` does not work if some elements of ``vector`` should be
    masked.  This performs a log_softmax on just the non-masked portions of ``vector``.  Passing
    ``None`` in for the mask is also acceptable; you'll just get a regular log_softmax.
    ``vector`` can have an arbitrary number of dimensions; the only requirement is that ``mask`` is
    broadcastable to ``vector's`` shape.  If ``mask`` has fewer dimensions than ``vector``, we will
    unsqueeze on dimension 1 until they match.  If you need a different unsqueezing of your mask,
    do it yourself before passing the mask into this function.
    In the case that the input vector is completely masked, the return value of this function is
    arbitrary, but not ``nan``.  You should be masking the result of whatever computation comes out
    of this in that case, anyway, so the specific values returned shouldn't matter.  Also, the way
    that we deal with this case relies on having single-precision floats; mixing half-precision
    floats with fully-masked vectors will likely give you ``nans``.
    If your logits are all extremely negative (i.e., the max value in your logit vector is -50 or
    lower), the way we handle masking here could mess you up.  But if you've got logit values that
    extreme, you've got bigger problems than this.
    """
    if mask is not None:
        mask = mask.float()
        while mask.dim() < vector.dim():
            mask = mask.unsqueeze(1)
        # vector + mask.log() is an easy way to zero out masked elements in logspace, but it
        # results in nans when the whole vector is masked.  We need a very small value instead of a
        # zero in the mask for these cases.  log(1 + 1e-45) is still basically 0, so we can safely
        # just add 1e-45 before calling mask.log().  We use 1e-45 because 1e-46 is so small it
        # becomes 0 - this is just the smallest value we can actually use.
        vector = vector + (mask + 1e-45).log()
    return torch.nn.functional.log_softmax(vector, dim=dim)


def concat_padded_tensors(t1: torch.Tensor, t2: torch.Tensor, pad_token: int = 0) -> torch.Tensor:
    assert len(t1.shape) == len(t2.shape)
    assert all(d1 == d2 for d1, d2 in zip(t1.shape[:-1], t2.shape[:-1]))
    concat_dim = t1.shape[-1] + t2.shape[-1]

    v1 = t1.reshape(-1, t1.shape[-1])
    v2 = t2.reshape(-1, t2.shape[-1])
    out = torch.cat([v1, v2], dim=-1)
    # out = v1.new_full((v1.shape[0], concat_dim), pad_token)
    # out[:, : v1.shape[1]] = v1
    v1_lens = (v1 != pad_token).int().sum(-1)
    scatter_index = torch.cat([v1_lens.unsqueeze(-1), v2.new_ones((v2.shape[0], v2.shape[1] - 1))], dim=-1)
    scatter_index = torch.cumsum(scatter_index, dim=-1)
    out.scatter_(-1, scatter_index, v2)

    return out.reshape(*t1.shape[:-1], concat_dim)


def pad_tensor(tensor_to_pad: torch.Tensor, new_size: int = 0, pad_token: int = 0) -> torch.Tensor:
    if tensor_to_pad.shape[-1] >= new_size:
        return tensor_to_pad

    padded_tensor = tensor_to_pad.new_full(tensor_to_pad.shape[:-1] + (new_size,), pad_token)
    padded_tensor[..., : tensor_to_pad.shape[-1]] = tensor_to_pad
    return padded_tensor


def tanh_clip(x, clip_val=None):
    if clip_val is not None:
        return clip_val * torch.tanh((1.0 / clip_val) * x)
    else:
        return x


def calc_nce_regulaizer(scores, regularizer_coef=4e-2):
    return regularizer_coef * (scores ** 2.0).mean()


class InfoNCE(nn.Module):
    def __init__(
        self,
        model,
        pad_token_id: int,
        inbatch_negatives: bool = False,
        demi: bool = False,
        encoder_emb_method: str = "first_token",
        clip_val: float = 100.0,
        project: Optional[str] = None,
    ):
        """
        """
        super().__init__()
        self.model = model
        self.pad_token_id = pad_token_id
        self.inbatch_negatives = inbatch_negatives
        self.demi = demi
        self.encoder_emb_method = encoder_emb_method
        self.clip_val = clip_val
        if project is None:
            self.mlp = None
        elif project == "linear":
            self.mlp = nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size)
        else:
            self.mlp = nn.Sequential(
                nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size), nn.ReLU()
            )

    def forward(
        self,
        history_input_ids,
        positive_input_ids,
        negative_input_ids,
        history_token_type_ids=None,
        positive_token_type_ids=None,
        negative_token_type_ids=None,
        history_attention_mask=None,
        positive_attention_mask=None,
        negative_attention_masks=None,
    ):
        assert (
            history_input_ids.shape[0] == positive_input_ids.shape[0]
        ), "history_ids and positive_ids must share the first dim"
        assert (
            history_input_ids.shape[0] == negative_input_ids.shape[0]
        ), "history_ids and negative_ids must share the first dim"

        B = history_input_ids.shape[0]

        if self.model.config.is_encoder_decoder:
            candidates, history_hidden_states = self._get_queries_and_candidates_encoder_decoder(
                history_input_ids,
                history_attention_mask,
                positive_input_ids,
                positive_attention_mask,
                negative_input_ids,
                negative_attention_masks,
            )
        else:
            candidates, history_hidden_states = self._get_queries_and_candidates_decoder_only(
                history_input_ids,
                history_attention_mask,
                history_token_type_ids,
                positive_input_ids,
                positive_attention_mask,
                positive_token_type_ids,
                negative_input_ids,
                negative_attention_masks,
                negative_token_type_ids,
            )

        negative_mask = (negative_input_ids != self.pad_token_id).sum(-1).sum(-1) > 0
        H = history_hidden_states.shape[-1]

        scores = torch.bmm(candidates, history_hidden_states.unsqueeze(1).transpose(1, 2)).squeeze(-1) / np.sqrt(H)

        easy_nce_loss = None
        if self.demi or self.inbatch_negatives:
            inbatch_mask = (
                (1 - torch.eye(B, device=scores.device)).unsqueeze(-1).expand(B, B, candidates.shape[1]).reshape(B, -1)
            )
            mask = torch.cat([torch.ones_like(scores), inbatch_mask], dim=-1)

            inbatch_scores = torch.mm(history_hidden_states, candidates.view(-1, candidates.shape[-1]).T) / np.sqrt(H)

            if self.demi:
                easy_scores = masked_log_softmax(inbatch_scores, inbatch_mask)
                if self.stabilize:
                    easy_scores = tanh_clip(easy_scores, self.clip_val)
                    easy_reg = calc_nce_regulaizer(easy_scores, self.regularizer_coef)
                else:
                    easy_reg = 0.0
                easy_nce_loss = -easy_scores[negative_mask, 0].mean() + easy_reg
                scores = F.log_softmax(scores, dim=-1)
            else:
                scores = masked_log_softmax(torch.cat([scores, inbatch_scores], dim=-1), mask)
        else:
            scores = F.log_softmax(scores, dim=-1)

        if self.stabilize:
            scores = tanh_clip(scores, self.clip_val)
            reg = calc_nce_regulaizer(scores, self.regularizer_coef)
        else:
            reg = 0.0

        _, max_score_indices = torch.max(scores, dim=1)
        selected_cands = max_score_indices

        hits_at_1 = (max_score_indices[negative_mask] == 0).float().sum() / negative_mask.int().sum()
        nce_loss = -scores[negative_mask, 0].mean() + reg

        return InfoNCEOutput(
            scores,
            selected_cands,
            nce_loss,
            hits_at_1,
            easy_nce_loss,
        )

    def _get_queries_and_candidates_decoder_only(
        self,
        history_input_ids,
        history_attention_mask,
        history_token_type_ids,
        positive_input_ids,
        positive_attention_mask,
        positive_token_type_ids,
        negative_input_ids,
        negative_attention_masks,
        negative_token_type_ids,
    ):
        B, Lh = history_input_ids.shape
        batch_range = torch.arange(B)

        history_lengths = (history_input_ids != self.pad_token_id).sum(-1)

        history_output = self.model(
            input_ids=history_input_ids,
            token_type_ids=history_token_type_ids,
            attention_mask=history_attention_mask,
            output_hidden_states=True,
        )
        history_hidden_states = history_output.hidden_states[-1][batch_range, history_lengths - 1]
        H = history_hidden_states.shape[-1]
        pos_model_input_ids = concat_padded_tensors(history_input_ids, positive_input_ids, self.pad_token_id)
        if history_token_type_ids is not None and positive_token_type_ids is not None:
            pos_model_token_types = concat_padded_tensors(
                history_token_type_ids, positive_token_type_ids, self.pad_token_id
            )
        else:
            pos_model_token_types = None
        if history_attention_mask is not None and positive_attention_mask is not None:
            pos_model_attn_mask = concat_padded_tensors(history_attention_mask, positive_attention_mask)
        else:
            pos_model_attn_mask = None
        positive_lengths = (pos_model_input_ids != self.pad_token_id).sum(-1)
        pos_output = self.model(
            input_ids=pos_model_input_ids,
            token_type_ids=pos_model_token_types,
            attention_mask=pos_model_attn_mask,
            output_hidden_states=True,
        )
        pos_hidden_states = pos_output.hidden_states[-1][batch_range, positive_lengths - 1]
        N = negative_input_ids.shape[1]
        neg_model_input_ids = concat_padded_tensors(
            history_input_ids.unsqueeze(1).expand(B, N, Lh), negative_input_ids, self.pad_token_id
        )
        Ln = neg_model_input_ids.shape[-1]
        if history_token_type_ids is not None and negative_token_type_ids is not None:
            neg_model_token_types = concat_padded_tensors(
                history_token_type_ids.unsqueeze(1).expand(B, N, Lh),
                negative_token_type_ids,
                self.pad_token_id,
            )
        else:
            neg_model_token_types = None
        if history_attention_mask is not None and negative_attention_masks is not None:
            neg_model_attn_mask = concat_padded_tensors(
                history_attention_mask.unsqueeze(1).expand(B, N, Lh),
                negative_attention_masks,
                self.pad_token_id,
            )
        else:
            neg_model_attn_mask = None
        negative_lengths = (neg_model_input_ids != self.pad_token_id).sum(-1).view(-1)
        neg_output = self.model(
            input_ids=neg_model_input_ids,
            token_type_ids=neg_model_token_types,
            attention_mask=neg_model_attn_mask,
            output_hidden_states=True,
        )
        neg_hidden_states = neg_output.hidden_states[-1].view(-1, Ln, H)[torch.arange(B * N), negative_lengths - 1]
        if self.mlp is not None:
            pos_hidden_states = self.mlp(pos_hidden_states)
            neg_hidden_states = self.mlp(neg_hidden_states)
            history_hidden_states = self.mlp(history_hidden_states)

        candidates = torch.cat([pos_hidden_states.unsqueeze(1), neg_hidden_states.view(B, N, H)], dim=1)
        return candidates, history_hidden_states

    def _get_queries_and_candidates_encoder_decoder(
        self,
        history_input_ids,
        history_attention_mask,
        positive_input_ids,
        positive_attention_mask,
        negative_input_ids,
        negative_attention_masks,
    ):
        B, Lh = history_input_ids.shape
        N = negative_input_ids.shape[1]

        history_output = self.model(
            input_ids=history_input_ids,
            attention_mask=history_attention_mask,
            decoder_input_ids=positive_input_ids[:, 0].unsqueeze(-1),
            decoder_attention_mask=positive_attention_mask[:, 0].unsqueeze(-1),
            output_hidden_states=True,
        )

        if self.encoder_emb_method == "mean_pool":
            history_mask = (
                (history_input_ids != self.pad_token_id)
                .int()
                .unsqueeze(-1)
                .expand_as(history_output.encoder_last_hidden_state)
            )
            history_lengths = (history_input_ids != self.pad_token_id).sum(-1).unsqueeze(-1)
            history_hidden_states = (
                torch.sum(history_output.encoder_last_hidden_state * history_mask, dim=1) / history_lengths
            )
        elif self.encoder_emb_method == "dec_first":
            history_hidden_states = history_output.decoder_hidden_states[-1][:, 0]
        else:
            history_hidden_states = history_output.encoder_last_hidden_state[:, 0]

        Lp = positive_input_ids.shape[-1]
        Ln = negative_input_ids.shape[-1]

        if Ln > Lp:
            positive_input_ids = pad_tensor(positive_input_ids, negative_input_ids.shape[-1], self.pad_token_id)
            positive_attention_mask = pad_tensor(positive_attention_mask, negative_input_ids.shape[-1])
        elif Lp > Ln:
            negative_input_ids = pad_tensor(negative_input_ids, positive_input_ids.shape[-1], self.pad_token_id)
            negative_attention_masks = pad_tensor(negative_attention_masks, positive_input_ids.shape[-1])

        decoder_input_ids = torch.cat(
            [positive_input_ids.unsqueeze(1), negative_input_ids],
            dim=1,
        ).view(B * (N + 1), -1)
        decoder_attention_mask = torch.cat(
            [positive_attention_mask.unsqueeze(1), negative_attention_masks], dim=1
        ).view(B * (N + 1), -1)

        decoder_lengths = (decoder_input_ids != self.pad_token_id).sum(-1)

        output = self.model(
            input_ids=history_input_ids.unsqueeze(1).expand(B, N + 1, Lh).reshape(-1, Lh),
            attention_mask=history_attention_mask.unsqueeze(1).expand(B, N + 1, Lh).reshape(-1, Lh),
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            output_hidden_states=True,
        )

        candidates = output.decoder_hidden_states[-1][torch.arange(B * (N + 1)), decoder_lengths - 1]
        if self.mlp is not None:
            candidates = self.mlp(candidates)
            history_hidden_states = self.mlp(history_hidden_states)

        return candidates.view(B, N + 1, -1), history_hidden_states
