# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch LLaMA model."""

import math
import warnings
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)
from transformers.models.llama.configuration_llama import LlamaConfig
from flash_attn import flash_attn_func, flash_attn_varlen_func
from flash_attn.bert_padding import (
    index_first_axis,
    index_put_first_axis,
    rearrange,
    pad_input,
    unpad_input,
)
from transformers.modeling_flash_attention_utils import _upad_input
from transformers.models.llama.modeling_llama import (
    rotate_half,
    apply_rotary_pos_emb,
    repeat_kv,
    _CONFIG_FOR_DOC,
    ALL_LAYERNORM_LAYERS,
    LlamaMLP,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
    LlamaLinearScalingRotaryEmbedding,
    LlamaDynamicNTKScalingRotaryEmbedding,
)


logger = logging.get_logger(__name__)


def _get_unpad_data(attention_mask):
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


class LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(config, "head_dim", self.hidden_size // self.num_heads)
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)

        # TODO (joao): remove in v4.45 (RoPE is computed in the model, not in the decoder layers)
        self.rotary_emb = LlamaRotaryEmbedding(config=self.config)

    def forward(self, *args, **kwargs):
        raise NotImplementedError


class LlamaFlashAttention2(LlamaAttention):
    """
    Llama flash attention module. This module inherits from `LlamaAttention` as the weights of the module stays
    untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
    flash attention and deal with padding tokens in case the input contains any of them.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # TODO: Should be removed once Flash Attention for RoCm is bumped to 2.1.
        # flash_attn<2.1 generates top-left aligned causal mask, while what is needed here is bottom-right alignement, that was made default for flash_attn>=2.1. This attribute is used to handle this difference. Reference: https://github.com/Dao-AILab/flash-attention/releases/tag/v2.1.0.
        # Beware that with flash_attn<2.1, using q_seqlen != k_seqlen (except for the case q_seqlen == 1) produces a wrong mask (top-left).
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        cu_seqlens: torch.LongTensor,
        max_seqlen_in_batch: int,
        position_ids: torch.LongTensor,
    ) -> torch.Tensor:
        bsz, q_len, hidden_size = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Flash attention requires the input to have the shape
        # batch_size x seq_length x head_dim x hidden_dim
        # therefore we just need to keep the original shape
        query_states = query_states.view(
            bsz, q_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        # TODO: These transpose are quite inefficient but Flash Attention requires the layout [batch_size, sequence_length, num_heads, head_dim]. We would need to refactor the KV cache
        # to be able to avoid many of these transpose/reshape/view.
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        dropout_rate = self.attention_dropout if self.training else 0.0

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in the correct dtype just to be sure everything works as expected.
        # This might slowdown training & inference so it is recommended to not cast the LayerNorms
        # in fp32. (GemmaRMSNorm handles it correctly)

        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            # Handle the case where the model is quantized
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        attn_output = self._flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            cu_seqlens=cu_seqlens,
            max_seqlen_in_batch=max_seqlen_in_batch,
            dropout=dropout_rate,
        )

        attn_output = attn_output.reshape(bsz, q_len, hidden_size).contiguous()
        attn_output = self.o_proj(attn_output)

        # return attn_output, attn_weights, past_key_value
        return attn_output, None, None

    def _flash_attention_forward(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        cu_seqlens,
        max_seqlen_in_batch,
        dropout=0.0,
        softmax_scale=None,
    ):
        """
        Calls the forward method of Flash Attention - if the input hidden states contain at least one padding token
        first unpad the input, then computes the attention scores and pad the final attention scores.

        Args:
            query_states (`torch.Tensor`):
                Input query states to be passed to Flash Attention API
            key_states (`torch.Tensor`):
                Input key states to be passed to Flash Attention API
            value_states (`torch.Tensor`):
                Input value states to be passed to Flash Attention API
            dropout (`float`):
                Attention dropout
            softmax_scale (`float`, *optional*):
                The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim)
        """
        _, query_length, _, _ = query_states.shape
        batch_size, kv_seq_len, num_key_value_heads, head_dim = key_states.shape

        if attention_mask is None:
            key_states = key_states.reshape(
                batch_size * kv_seq_len, num_key_value_heads, head_dim
            )
            value_states = value_states.reshape(
                batch_size * kv_seq_len, num_key_value_heads, head_dim
            )
            query_states = query_states.reshape(
                batch_size * query_length, self.num_heads, head_dim
            )
        else:
            query_length = query_states.shape[1]
            query_states, key_states, value_states, indices_q, _, _ = _upad_input(
                query_states, key_states, value_states, attention_mask, query_length
            )

        attn_output = flash_attn_varlen_func(
            query_states,
            key_states,
            value_states,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen_in_batch,
            max_seqlen_k=max_seqlen_in_batch,
            dropout_p=dropout,
            softmax_scale=softmax_scale,
            causal=True,
        )

        if attention_mask is not None:
            # we need to pad the output back to normal shape
            attn_output = pad_input(attn_output, indices_q, batch_size, query_length)

        return attn_output


class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        # NOTE: force using flash-attention 2
        self.self_attn = LlamaFlashAttention2(config=config, layer_idx=layer_idx)

        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        cu_seqlens: torch.LongTensor,
        max_seqlen_in_batch: int,
        position_ids: torch.LongTensor,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            cu_seqlens (`torch.Int32Tensor`): cumulative sequence lengths of documents in the batch`
            max_seqlen_in_batch (`int`): maximum sequence length in the batch
            position_ids (`torch.LongTensor`): position ids of the input tokens
        """
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, _, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            cu_seqlens=cu_seqlens,
            max_seqlen_in_batch=max_seqlen_in_batch,
            position_ids=position_ids,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        return outputs


class LlamaPreTrainedModel(PreTrainedModel):
    config_class = LlamaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LlamaDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class LlamaModel(LlamaPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]

    Args:
        config: LlamaConfig
    """

    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        self.layers = nn.ModuleList(
            [
                LlamaDecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        # self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    # Ignore copy
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
        concat_mode: Optional[bool] = False,
        *args,
        **kwargs,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        # This special model cannot use cache or output attentions or hidden states
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if attention_mask is not None:
            attention_mask = attention_mask.bool()

            batch_size = attention_mask.shape[0]
            seqlen = attention_mask.shape[1]

            if inputs_embeds is None:
                if concat_mode:
                    # NOTE: in concat mode, we flatten the input ids and inputs_embeds
                    cu_input_ids = input_ids[attention_mask].unsqueeze(0)
                    cu_inputs_embeds = self.embed_tokens(cu_input_ids)
                else:
                    # NOTE: in non-concat mode, we keep the input ids and inputs_embeds as is
                    cu_inputs_embeds = self.embed_tokens(input_ids)
            else:
                if concat_mode:
                    # NOTE: in concat mode, we flatten the input ids and inputs_embeds
                    cu_inputs_embeds = inputs_embeds[attention_mask].unsqueeze(0)
                else:
                    cu_inputs_embeds = inputs_embeds

            # NOTE: We need to compute the unpad indices, cu_seqlens, and max_seqlen_in_batch to handle padding tokens
            unpad_indices, unpad_cu_seqlens, max_seqlen_in_batch = _get_unpad_data(
                attention_mask
            )

            # NOTE: if cu_seqlens is not provided, we use the attention mask to compute cu_seqlens
            # If it is provided, we use the provided cu_seqlens instead of unpad_cu_seqlens!
            if cu_seqlens is None:
                cu_seqlens = unpad_cu_seqlens

            if position_ids is None:
                position_ids = torch.arange(0, seqlen, device=cu_inputs_embeds.device)
                position_ids = position_ids.unsqueeze(0).expand(batch_size, seqlen)
            
            # NOTE: in concat mode, we flatten the position ids
            if concat_mode:
                cu_position_ids = position_ids[attention_mask].unsqueeze(0)
            else:
                cu_position_ids = position_ids

            # Batch size 1 embed positions
            hidden_states = cu_inputs_embeds
            position_ids = cu_position_ids
        else:
            # NOTE: If attention mask is not provided, we assume no padding
            # If there is any padding and attention mask is not provided, we will not be able to handle padding tokens
            # This would lead to incorrect results!
            batch_size, seqlen = input_ids.shape
            max_seqlen_in_batch = seqlen
            hidden_states = self.embed_tokens(input_ids)
            position_ids = (
                position_ids
                if position_ids is not None
                else torch.arange(seqlen, device=hidden_states.device)
                .unsqueeze(0)
                .expand(batch_size, seqlen)
            )

            # if attention mask and cu_seqlens are not provided, we assume no padding
            if cu_seqlens is None:
                cu_seqlens = torch.tensor(
                    [seqlen] * batch_size,
                    device=hidden_states.device,
                    dtype=torch.int32,
                )
                cu_seqlens = torch.cumsum(cu_seqlens, dim=0)

        # decoder layers
        for decoder_layer in self.layers:
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    cu_seqlens,
                    max_seqlen_in_batch,
                    position_ids,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    cu_seqlens=cu_seqlens,
                    max_seqlen_in_batch=max_seqlen_in_batch,
                    position_ids=position_ids,
                )

            hidden_states = layer_outputs[0]

        hidden_states = self.norm(hidden_states)

        # NOTE: If concat_mode is True, we need to recover the hidden states for batch size
        if concat_mode:
            # recover hidden states for batch size
            hidden_states = index_put_first_axis(
                hidden_states.squeeze(0), unpad_indices, batch_size * seqlen
            )
            hidden_states = rearrange(
                hidden_states, "(b s) ... -> b s ...", b=batch_size
            )

        if not return_dict:
            return tuple(v for v in [hidden_states, None, None, None] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
        )


class LlamaForCausalLM(LlamaPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        # self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
        concat_mode: Optional[bool] = False,
        *args,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            cu_seqlens=cu_seqlens,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            return_dict=return_dict,
            concat_mode=concat_mode,
        )
        hidden_states = outputs[0]

        if labels is not None:
            hidden_states = hidden_states[..., :-1, :]
            shift_labels = labels[..., 1:].contiguous()

            # flattened and only consider non-ignored labeled tokens
            hidden_states = hidden_states[shift_labels != -100]
            select_logits = self.lm_head(hidden_states).float()
            select_labels = shift_labels[shift_labels != -100]

            loss_fct = CrossEntropyLoss()

            shift_labels = shift_labels.to(select_logits.device)
            loss = loss_fct(select_logits, select_labels)
            logits = None
        else:
            loss = None
            logits = self.lm_head(hidden_states)
            logits = logits.float()

        # Do not return the logits for efficiency
        if not return_dict:
            return (loss,)

        return CausalLMOutputWithPast(loss=loss, logits=logits)
