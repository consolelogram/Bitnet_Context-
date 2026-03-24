"""
modeling_bitnet.py
Reconstructed from forensic audit of microsoft/bitnet-b1.58-2B-4T.

Confirmed architecture:
  - LLaMA-style transformer
  - relu² activation (not SiLU)
  - Sub-norm after attention and FFN (learned per-channel gain)
  - GQA: 20 query heads / 5 KV heads
  - RoPE theta=500000, head_dim=128
  - No biases on linear layers
  - RMSNorm throughout
  - tie_word_embeddings=True

Drop alongside configuration_bitnet.py.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union, List

from transformers import PreTrainedModel
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast, BaseModelOutputWithPast
from transformers.cache_utils import Cache, DynamicCache

from configuration_bitnet import BitNetConfig


# ─────────────────────────────────────────────────────────────────────────────
# Primitives
# ─────────────────────────────────────────────────────────────────────────────

class BitNetRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps    = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).to(x.dtype) * self.weight


class BitNetSubNorm(nn.Module):
    """
    Learned per-channel gain applied after attention/FFN output.
    Forensic finding: gain escalates from ~1x at layer 0 to ~9x at layer 29.
    Acts as the learned dequantization compensator.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * self.weight


def relu2(x: torch.Tensor) -> torch.Tensor:
    """Squared ReLU — confirmed from config hidden_act=relu2."""
    return F.relu(x).pow(2)


# ─────────────────────────────────────────────────────────────────────────────
# RoPE
# ─────────────────────────────────────────────────────────────────────────────

class BitNetRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_position_embeddings: int, theta: float):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.theta = theta
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x, position_ids):
        inv_freq = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        pos      = position_ids[:, None, :].float()
        freqs    = (inv_freq @ pos).transpose(1, 2)
        emb      = torch.cat([freqs, freqs], dim=-1)
        return emb.cos().to(x.dtype), emb.sin().to(x.dtype)


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin


# ─────────────────────────────────────────────────────────────────────────────
# MLP
# ─────────────────────────────────────────────────────────────────────────────

class BitNetMLP(nn.Module):
    def __init__(self, config: BitNetConfig):
        super().__init__()
        self.gate_proj    = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj      = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj    = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.ffn_sub_norm = BitNetSubNorm(config.intermediate_size)

    def forward(self, x):
        # SwiGLU-style with relu² gate; sub_norm on gated intermediate
        hidden = self.ffn_sub_norm(relu2(self.gate_proj(x)) * self.up_proj(x))
        return self.down_proj(hidden)


# ─────────────────────────────────────────────────────────────────────────────
# Attention
# ─────────────────────────────────────────────────────────────────────────────

class BitNetAttention(nn.Module):
    def __init__(self, config: BitNetConfig, layer_idx: int):
        super().__init__()
        self.layer_idx     = layer_idx
        self.hidden_size   = config.hidden_size
        self.num_heads     = config.num_attention_heads
        self.num_kv_heads  = config.num_key_value_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.head_dim      = config.hidden_size // self.num_heads
        self.scaling       = self.head_dim ** -0.5

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads    * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size,    bias=False)

        self.attn_sub_norm = BitNetSubNorm(self.num_heads * self.head_dim)

        self.rotary_emb = BitNetRotaryEmbedding(
            dim=self.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            theta=config.rope_theta,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        B, seq_len, _ = hidden_states.shape

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q = q.view(B, seq_len, self.num_heads,    self.head_dim).transpose(1, 2)
        k = k.view(B, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        if position_embeddings is not None:
            cos, sin = position_embeddings
        else:
            cos, sin = self.rotary_emb(q, position_ids)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if past_key_value is not None:
            cache_kwargs = {"cache_position": cache_position}
            k, v = past_key_value.update(k, v, self.layer_idx, cache_kwargs)

        # GQA expand
        k = k.repeat_interleave(self.num_kv_groups, dim=1)
        v = v.repeat_interleave(self.num_kv_groups, dim=1)

        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=(attention_mask is None),
            scale=self.scaling,
        )

        # sub_norm before output projection
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, seq_len, -1)
        attn_out = self.attn_sub_norm(attn_out)
        attn_out = self.o_proj(attn_out)

        return attn_out, None, past_key_value


# ─────────────────────────────────────────────────────────────────────────────
# Decoder layer
# ─────────────────────────────────────────────────────────────────────────────

class BitNetDecoderLayer(nn.Module):
    def __init__(self, config: BitNetConfig, layer_idx: int):
        super().__init__()
        self.self_attn          = BitNetAttention(config, layer_idx)
        self.mlp                = BitNetMLP(config)
        self.input_layernorm    = BitNetRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = BitNetRMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        # Attention block
        residual     = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _, past_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )
        hidden_states = residual + hidden_states

        # MLP block
        residual      = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, None, past_key_value


# ─────────────────────────────────────────────────────────────────────────────
# Base model
# ─────────────────────────────────────────────────────────────────────────────

class BitNetModel(PreTrainedModel):
    config_class = BitNetConfig

    def __init__(self, config: BitNetConfig):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers       = nn.ModuleList([
            BitNetDecoderLayer(config, i) for i in range(config.num_hidden_layers)
        ])
        self.norm         = BitNetRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.rotary_emb   = BitNetRotaryEmbedding(
            dim=config.hidden_size // config.num_attention_heads,
            max_position_embeddings=config.max_position_embeddings,
            theta=config.rope_theta,
        )
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ):
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds
        B, seq_len, _ = hidden_states.shape

        if position_ids is None:
            position_ids = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0)

        if cache_position is None:
            cache_position = torch.arange(seq_len, device=hidden_states.device)

        if past_key_values is None and use_cache:
            past_key_values = DynamicCache()

        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for layer in self.layers:
            hidden_states, _, past_key_values = layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=False,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )

        hidden_states = self.norm(hidden_states)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=None,
            attentions=None,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Causal LM head
# ─────────────────────────────────────────────────────────────────────────────

class BitNetForCausalLM(PreTrainedModel, GenerationMixin):
    config_class                    = BitNetConfig
    _tied_weights_keys              = ["lm_head.weight"]
    _keys_to_ignore_on_load_missing = [r"lm_head\.weight"]

    def __init__(self, config: BitNetConfig):
        super().__init__(config)
        self.model   = BitNetModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, value):
        self.lm_head = value

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            cache_position=cache_position,
        )

        logits = self.lm_head(outputs.last_hidden_state)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None,
        inputs_embeds=None, cache_position=None, **kwargs
    ):
        if past_key_values is not None:
            input_ids = input_ids[:, cache_position]

        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "attention_mask": attention_mask,
            "cache_position": cache_position,
            "use_cache": True,
        }
