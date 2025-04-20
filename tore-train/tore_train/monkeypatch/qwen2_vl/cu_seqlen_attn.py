import os
import inspect
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModel
from transformers.utils import (
    logging,
    is_flash_attn_greater_or_equal,
    is_flash_attn_2_available,
)
from transformers.models.qwen2_vl.modeling_qwen2_vl import (
    Cache,
    Qwen2VLAttention,
    Qwen2VLFlashAttention2,
    apply_multimodal_rotary_pos_emb,
    repeat_kv,
)

logger = logging.get_logger(__name__)

# Check if flash attention is available
if is_flash_attn_2_available():
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa
    from flash_attn import flash_attn_func, flash_attn_varlen_func

    _flash_supports_window_size = "window_size" in list(
        inspect.signature(flash_attn_func).parameters
    )


def get_cu_seqlens(position_ids: torch.Tensor) -> torch.Tensor:
    """
    Converts position IDs into cumulative sequence lengths by identifying the boundaries
    between different sequences in the batch. Sequences are identified by position IDs
    that reset to 0, indicating the start of a new sequence.

    Arguments:
        position_ids (`torch.Tensor`):
            Integer tensor containing position indices. Shape can be either:
            - (batch_size, sequence_length)
            - (total_sequence_length,) if already flattened
            Position IDs typically start from 0 and increment for each token in a sequence.
            A reset to 0 indicates the start of a new sequence.

    Returns:
        cu_seq_lens (`torch.Tensor`):
            Integer tensor containing the cumulative sequence lengths.
            Shape is (num_sequences + 1,), where num_sequences is the number of
            sequences identified in the batch (number of zeros in position_ids + 1).
            The first element is always 0, and the last element is the total sequence length.

    Example:
        >>> position_ids = torch.tensor([[0, 1, 2, 3, 0, 1, 2]])  # Two sequences: [0,1,2,3] and [0,1,2]
        >>> get_cu_seqlens(position_ids)
        tensor([0, 4, 7], dtype=torch.int32)  # Indicates sequences at indices 0-3 and 4-6

    Note:
        This function is typically used in conjunction with Flash Attention to handle
        variable-length sequences in a batch efficiently by indicating where each
        sequence starts and ends in the flattened tensor.
    """
    position_ids = position_ids.flatten()
    indices_q = torch.arange(
        position_ids.size(0), device=position_ids.device, dtype=torch.int32
    )
    cu_seq_lens = torch.cat(
        (
            indices_q[position_ids == 0],
            torch.tensor(
                position_ids.size(), device=position_ids.device, dtype=torch.int32
            ),
        )
    )
    return cu_seq_lens


def fa_peft_integration_check(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    target_dtype: Optional[torch.dtype] = None,
):
    """
    PEFT usually casts the layer norms in float32 for training stability reasons
    therefore the input hidden states gets silently casted in float32. Hence, we need
    cast them back in float16 / bfloat16 just to be sure everything works as expected.
    This might slowdown training & inference so it is recommended to not cast the LayerNorms!

    Args:
        query (`torch.Tensor`):
            Input query states to be passed to Flash Attention API
        key (`torch.Tensor`):
            Input key states to be passed to Flash Attention API
        value (`torch.Tensor`):
            Input value states to be passed to Flash Attention API
        target_dtype (`torch.dtype`, *optional*):
            The dtype to convert the attention tensors to. Conversion can be ignored by
            not providing the target dtype.
    """
    if target_dtype is None:
        return query, key, value

    input_dtype = value.dtype
    if input_dtype == torch.float32:
        logger.warning_once(
            f"The input hidden states seems to be silently casted in float32, this might be related to"
            f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
            f" {target_dtype}."
        )

        query = query.to(target_dtype)
        key = key.to(target_dtype)
        value = value.to(target_dtype)

    return query, key, value


flash_241 = is_flash_attn_greater_or_equal("2.4.1")
deterministic_g = os.environ.get("FLASH_ATTENTION_DETERMINISTIC", "0") == "1"


def _flash_attention_forward(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: torch.Tensor,
    query_length: int,
    is_causal: bool,
    dropout: float = 0.0,
    position_ids: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    sliding_window: Optional[int] = None,
    use_top_left_mask: bool = False,
    softcap: Optional[float] = None,
    deterministic: bool = None,
    cu_seq_lens_q: Optional[torch.LongTensor] = None,
    cu_seq_lens_k: Optional[torch.LongTensor] = None,
    max_length_q: Optional[int] = None,
    max_length_k: Optional[int] = None,
    target_dtype: Optional[torch.dtype] = None,
    **kwargs,
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
        attention_mask (`torch.Tensor`):
            The padding mask - corresponds to a tensor of size `(batch_size, seq_len)` where 0 stands for the
            position of padding tokens and 1 for the position of non-padding tokens.
        dropout (`float`):
            Attention dropout
        softmax_scale (`float`, *optional*):
            The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim)
        use_top_left_mask (`bool`, defaults to `False`):
            flash_attn<2.1 generates top-left aligned causal mask, while what is needed here is bottom-right alignement, that was made default for flash_attn>=2.1. This attribute is used to handle this difference.
        softcap (`float`, *optional*):
            Softcap for the attention logits, used e.g. in gemma2.
        deterministic (`bool`, *optional*):
            Determines if the deterministic option introduced in flash_attn>=2.4.1 is enabled.
    """
    if not use_top_left_mask:
        causal = is_causal
    else:
        # TODO: Remove the `query_length != 1` check once Flash Attention for RoCm is bumped to 2.1.
        causal = is_causal and query_length != 1

    # Assuming 4D tensors, key_states.shape[1] is the key/value sequence length (source length).
    use_sliding_windows = (
        _flash_supports_window_size
        and sliding_window is not None
        and key_states.shape[1] > sliding_window
    )
    flash_kwargs = (
        {"window_size": (sliding_window, sliding_window)} if use_sliding_windows else {}
    )

    if flash_241:
        if deterministic is None:
            deterministic = deterministic_g
        flash_kwargs["deterministic"] = deterministic

    if softcap is not None:
        flash_kwargs["softcap"] = softcap

    # PEFT possibly silently casts tensors to fp32, this potentially reconverts to correct dtype or is a no op
    query_states, key_states, value_states = fa_peft_integration_check(
        query_states, key_states, value_states, target_dtype
    )

    batch_size = query_states.size(0)

    if cu_seq_lens_q is None or cu_seq_lens_k is None:
        query_states = query_states.view(-1, query_states.size(-2), query_states.size(-1))
        key_states = key_states.contiguous().view(-1, key_states.size(-2), key_states.size(-1))
        value_states = value_states.contiguous().view(-1, value_states.size(-2), value_states.size(-1))
       
        # Extract the cu_seq_lens and max_length from position_ids
        cu_seq_lens_q = get_cu_seqlens(position_ids)
        max_length = position_ids.max() + 1

        cu_seq_lens_q, cu_seq_lens_k = cu_seq_lens_q, cu_seq_lens_q
        max_length_q, max_length_k = max_length, max_length

    else:
        query_states = query_states.reshape(
            -1, query_states.size(-2), query_states.size(-1)
        )
        key_states = key_states.reshape(-1, key_states.size(-2), key_states.size(-1))
        value_states = value_states.reshape(
            -1, value_states.size(-2), value_states.size(-1)
        )

    attn_output = flash_attn_varlen_func(
        query_states,
        key_states,
        value_states,
        cu_seqlens_q=cu_seq_lens_q,
        cu_seqlens_k=cu_seq_lens_k,
        max_seqlen_q=max_length_q,
        max_seqlen_k=max_length_k,
        dropout_p=dropout,
        softmax_scale=softmax_scale,
        causal=causal,
        **flash_kwargs,
    )

    attn_output = attn_output.view(
        batch_size, -1, attn_output.size(-2), attn_output.size(-1)
    )

    return attn_output


class Qwen2VLCuSeqLenAttn(Qwen2VLAttention):
    """
    This class is used for document integrity attention.
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[
            Tuple[torch.Tensor, torch.Tensor]
        ] = None,  # necessary, but kept here for BC
    ):
        assert position_ids is not None, "position_ids is required for document integrity attention"
        assert past_key_value is None, "past_key_value is not supported for document integrity attention"
        # We use position_ids for document integrity attention, so no attention mask is needed
        attention_mask = None

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

        # Because the input can be padded, the absolute sequence length depends on the max position id.
        cos, sin = position_embeddings
        query_states, key_states = apply_multimodal_rotary_pos_emb(
            query_states, key_states, cos, sin, self.rope_scaling["mrope_section"]
        )

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        dropout_rate = 0.0 if not self.training else self.attention_dropout

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in float16 just to be sure everything works as expected.
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

        # Reashape to the expected shape for Flash Attention
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        if (
            self.config.use_sliding_window
            and getattr(self.config, "sliding_window", None) is not None
            and self.layer_idx >= self.config.max_window_layers
        ):
            sliding_window = self.config.sliding_window
        else:
            sliding_window = None

        attn_output = _flash_attention_forward(
            query_states=query_states,
            key_states=key_states,
            value_states=value_states,
            attention_mask=None,
            query_length=q_len,
            dropout=dropout_rate,
            sliding_window=sliding_window,
            is_causal=self.is_causal,
            use_top_left_mask=self._flash_attn_uses_top_left_mask,
            # positon_ids is expanded for mrope
            position_ids=position_ids[0].clone(),
        )

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


def apply_qwen2vl_cu_seqlen_attn(model: AutoModel = None):
    for name, module in model.named_modules():
        if isinstance(module, (Qwen2VLAttention, Qwen2VLFlashAttention2)):
            module.forward = Qwen2VLCuSeqLenAttn.forward.__get__(module, type(module))

    return model
