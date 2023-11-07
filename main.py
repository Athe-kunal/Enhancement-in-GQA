import torch

from grouped_query_attention_pytorch.attention import scaled_dot_product_gqa

# shapes: (batch_size, seq_len, num_heads, head_dim)
query = torch.randn(1, 256, 8, 64, device="cuda", dtype=torch.float16)
key = torch.randn(1, 128, 2, 64, device="cuda", dtype=torch.float16)
value = torch.randn(1, 128, 2, 64, device="cuda", dtype=torch.float16)

out, attn_weights = scaled_dot_product_gqa(
    query,
    key,
    value,
    is_causal=True,  # default: False
    need_weights=True,  # default: False, which returns 'attn_weights=None'
)
print(out.shape)  # (batch_size, q_seq_len, kv_heads, embed_dim)
# torch.Size([1, 256, 2, 64])
print(attn_weights.shape)  # (batch_size, q_seq_len, kv_seq_len, kv_heads)
# torch.Size([1, 256, 128, 2])