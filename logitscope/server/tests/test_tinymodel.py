import torch

from logitscope.tinymodel import ByteTokenizer, build_tiny_bundle


def test_forward_shape(tiny):
    ids = torch.tensor([[1, 2, 3, 4]])
    with torch.no_grad():
        out = tiny.model(input_ids=ids)
    assert out.logits.shape == (1, 4, 256)


def test_same_seed_same_weights():
    a = build_tiny_bundle(seed=0)
    b = build_tiny_bundle(seed=0)
    ids = torch.tensor([[5, 6, 7]])
    with torch.no_grad():
        la = a.model(input_ids=ids).logits
        lb = b.model(input_ids=ids).logits
    assert torch.equal(la, lb)


def test_byte_tokenizer_roundtrip():
    tok = ByteTokenizer()
    s = "hello scope"
    assert tok.decode(tok.encode(s)) == s
    assert all(0 <= i < 256 for i in tok.encode("héllo"))


def test_eager_attention_configured(tiny):
    # attention capture needs eager, sdpa never materializes the weights
    assert tiny.model.config._attn_implementation == "eager"
