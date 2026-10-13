"""tiny random llama for tests and dev. builds in-process in well under a second,
so the whole suite runs with zero downloads."""

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from .model import ModelBundle


class ByteTokenizer:
    """utf-8 bytes as token ids. vocab of 256 lines up exactly with the tiny config,
    and every string is tokenizable by construction."""

    vocab_size = 256
    eos_token_id = 0

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, ids: list[int]) -> str:
        return bytes(int(i) % 256 for i in ids).decode("utf-8", errors="replace")


def build_tiny_config() -> LlamaConfig:
    return LlamaConfig(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=1024,
        tie_word_embeddings=False,
        attn_implementation="eager",
    )


def build_tiny_bundle(seed: int = 0) -> ModelBundle:
    torch.manual_seed(seed)
    model = LlamaForCausalLM(build_tiny_config())
    model.eval()
    return ModelBundle(
        model_id="tiny-random",
        model=model,
        tokenizer=ByteTokenizer(),
        device=torch.device("cpu"),
    )
