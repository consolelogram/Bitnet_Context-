"""
configuration_bitnet.py
Reconstructed from forensic audit of microsoft/bitnet-b1.58-2B-4T.
Drop this file alongside modeling_bitnet.py in your working directory.
"""
from transformers import PretrainedConfig


class BitNetConfig(PretrainedConfig):
    model_type = "bitnet"

    def __init__(
        self,
        vocab_size=128256,
        hidden_size=2560,
        intermediate_size=6912,
        num_hidden_layers=30,
        num_attention_heads=20,
        num_key_value_heads=5,
        hidden_act="relu2",
        max_position_embeddings=4096,
        initializer_range=0.02,
        rms_norm_eps=1e-5,
        use_cache=True,
        bos_token_id=128000,
        eos_token_id=128001,
        rope_theta=500000.0,
        tie_word_embeddings=True,
        quantization_config=None,
        **kwargs,
    ):
        self.vocab_size              = vocab_size
        self.hidden_size             = hidden_size
        self.intermediate_size       = intermediate_size
        self.num_hidden_layers       = num_hidden_layers
        self.num_attention_heads     = num_attention_heads
        self.num_key_value_heads     = num_key_value_heads
        self.hidden_act              = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range       = initializer_range
        self.rms_norm_eps            = rms_norm_eps
        self.use_cache               = use_cache
        self.rope_theta              = rope_theta
        self.quantization_config     = quantization_config
        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
