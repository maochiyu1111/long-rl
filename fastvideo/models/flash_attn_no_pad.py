import torch
import torch.nn.functional as F
from einops import rearrange

try:
    from flash_attn import flash_attn_varlen_qkvpacked_func
    from flash_attn.bert_padding import pad_input, unpad_input

    _FLASH_ATTN_AVAILABLE = True
except ImportError:
    flash_attn_varlen_qkvpacked_func = None
    pad_input = None
    unpad_input = None
    _FLASH_ATTN_AVAILABLE = False

try:
    import torch_npu

    _TORCH_NPU_FLASH_ATTN_AVAILABLE = hasattr(torch_npu, "npu_fusion_attention")
except ImportError:
    torch_npu = None
    _TORCH_NPU_FLASH_ATTN_AVAILABLE = False

_NPU_CAUSAL_MASK_SIZE = 2048
_NPU_CAUSAL_MASK_CACHE = {}


def _normalize_key_padding_mask(key_padding_mask, *, batch_size, seqlen, device):
    if key_padding_mask is None:
        return torch.ones((batch_size, seqlen), dtype=torch.bool, device=device)
    return key_padding_mask.to(device=device, dtype=torch.bool)


def _unpad_varlen_input(hidden_states, attention_mask):
    if unpad_input is not None:
        x_unpad, indices, cu_seqlens, max_s, _ = unpad_input(hidden_states, attention_mask)
        return x_unpad, indices, cu_seqlens, max_s

    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.reshape(-1), as_tuple=False).flatten()
    max_s = int(seqlens_in_batch.max().item()) if seqlens_in_batch.numel() > 0 else 0
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    hidden_states = rearrange(hidden_states, "b s ... -> (b s) ...")
    return hidden_states.index_select(0, indices), indices, cu_seqlens, max_s


def _index_varlen_input(hidden_states, indices):
    hidden_states = rearrange(hidden_states, "b s ... -> (b s) ...")
    return hidden_states.index_select(0, indices)


def _pad_varlen_input(hidden_states, indices, batch_size, seqlen):
    if pad_input is not None:
        return pad_input(hidden_states, indices, batch_size, seqlen)

    output = hidden_states.new_zeros((batch_size * seqlen, *hidden_states.shape[1:]))
    if indices.numel() > 0:
        output.index_copy_(0, indices, hidden_states)
    return rearrange(output, "(b s) ... -> b s ...", b=batch_size)


def _get_npu_causal_mask(device):
    cache_key = (device.type, device.index)
    causal_mask = _NPU_CAUSAL_MASK_CACHE.get(cache_key)
    if causal_mask is None:
        causal_mask = torch.triu(
            torch.ones(
                (_NPU_CAUSAL_MASK_SIZE, _NPU_CAUSAL_MASK_SIZE),
                device=device,
                dtype=torch.bool,
            ),
            diagonal=1,
        )
        _NPU_CAUSAL_MASK_CACHE[cache_key] = causal_mask
    return causal_mask


def _flash_attn_cuda(qkv, key_padding_mask, causal, dropout_p, softmax_scale):
    batch_size = qkv.shape[0]
    seqlen = qkv.shape[1]
    nheads = qkv.shape[-2]
    x = rearrange(qkv, "b s three h d -> b s (three h d)")
    x_unpad, indices, cu_seqlens, max_s = _unpad_varlen_input(x, key_padding_mask)
    x_unpad = rearrange(x_unpad, "nnz (three h d) -> nnz three h d", three=3, h=nheads)
    output_unpad = flash_attn_varlen_qkvpacked_func(
        x_unpad,
        cu_seqlens,
        max_s,
        dropout_p,
        softmax_scale=softmax_scale,
        causal=causal,
    )
    output = rearrange(
        _pad_varlen_input(
            rearrange(output_unpad, "nnz h d -> nnz (h d)"),
            indices,
            batch_size,
            seqlen,
        ),
        "b s (h d) -> b s h d",
        h=nheads,
    )
    return output


def _flash_attn_npu(qkv, key_padding_mask, causal, dropout_p, softmax_scale):
    batch_size, seqlen, _, nheads, head_dim = qkv.shape
    query = qkv[:, :, 0].contiguous()
    key = qkv[:, :, 1].contiguous()
    value = qkv[:, :, 2].contiguous()

    query_unpad, indices, cu_seqlens, _ = _unpad_varlen_input(query, key_padding_mask)
    key_unpad = _index_varlen_input(key, indices)
    value_unpad = _index_varlen_input(value, indices)

    if query_unpad.shape[0] == 0:
        return qkv.new_zeros((batch_size, seqlen, nheads, head_dim))

    scale = softmax_scale if softmax_scale is not None else head_dim**-0.5
    npu_kwargs = {
        "head_num": nheads,
        "input_layout": "TND",
        "scale": scale,
        "keep_prob": 1.0 - dropout_p,
        "actual_seq_qlen": tuple(cu_seqlens[1:].cpu().tolist()),
        "actual_seq_kvlen": tuple(cu_seqlens[1:].cpu().tolist()),
    }
    if causal:
        npu_kwargs["atten_mask"] = _get_npu_causal_mask(query.device)
        npu_kwargs["sparse_mode"] = 3

    attn_out = torch_npu.npu_fusion_attention(
        query_unpad,
        key_unpad,
        value_unpad,
        **npu_kwargs,
    )[0]
    return _pad_varlen_input(attn_out, indices, batch_size, seqlen)


def _flash_attn_fallback(qkv, key_padding_mask, causal, dropout_p, softmax_scale):
    query = qkv[:, :, 0].permute(0, 2, 1, 3).contiguous()
    key = qkv[:, :, 1].permute(0, 2, 1, 3).contiguous()
    value = qkv[:, :, 2].permute(0, 2, 1, 3).contiguous()

    batch_size, nheads, seqlen, _ = query.shape
    valid_mask = _normalize_key_padding_mask(
        key_padding_mask,
        batch_size=batch_size,
        seqlen=seqlen,
        device=query.device,
    )
    attn_mask = valid_mask[:, None, None, :].expand(batch_size, nheads, seqlen, seqlen)
    if causal:
        causal_mask = torch.tril(torch.ones((seqlen, seqlen), device=query.device, dtype=torch.bool))
        attn_mask = attn_mask & causal_mask.unsqueeze(0).unsqueeze(0)

    sdpa_kwargs = {
        "attn_mask": attn_mask,
        "dropout_p": dropout_p,
        "is_causal": False,
    }
    if softmax_scale is not None:
        sdpa_kwargs["scale"] = softmax_scale

    output = F.scaled_dot_product_attention(query, key, value, **sdpa_kwargs)
    output = output * valid_mask[:, None, :, None].to(output.dtype)
    return output.permute(0, 2, 1, 3).contiguous()


def flash_attn_no_pad(
    qkv,
    key_padding_mask,
    causal=False,
    dropout_p=0.0,
    softmax_scale=None,
):
    device_type = qkv.device.type
    key_padding_mask = _normalize_key_padding_mask(
        key_padding_mask,
        batch_size=qkv.shape[0],
        seqlen=qkv.shape[1],
        device=qkv.device,
    )

    if device_type == "cuda" and _FLASH_ATTN_AVAILABLE:
        return _flash_attn_cuda(qkv, key_padding_mask, causal, dropout_p, softmax_scale)
    if device_type == "npu" and _TORCH_NPU_FLASH_ATTN_AVAILABLE:
        return _flash_attn_npu(qkv, key_padding_mask, causal, dropout_p, softmax_scale)
    return _flash_attn_fallback(qkv, key_padding_mask, causal, dropout_p, softmax_scale)
