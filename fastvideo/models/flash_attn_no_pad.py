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


def _normalize_key_padding_mask(key_padding_mask, *, batch_size, seqlen, device):
    if key_padding_mask is None:
        return torch.ones((batch_size, seqlen), dtype=torch.bool, device=device)
    return key_padding_mask.to(device=device, dtype=torch.bool)


def _flash_attn_cuda(qkv, key_padding_mask, causal, dropout_p, softmax_scale):
    batch_size = qkv.shape[0]
    seqlen = qkv.shape[1]
    nheads = qkv.shape[-2]
    x = rearrange(qkv, "b s three h d -> b s (three h d)")
    x_unpad, indices, cu_seqlens, max_s, _ = unpad_input(x, key_padding_mask)
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
        pad_input(rearrange(output_unpad, "nnz h d -> nnz (h d)"), indices, batch_size, seqlen),
        "b s (h d) -> b s h d",
        h=nheads,
    )
    return output


def _flash_attn_npu(qkv, key_padding_mask, causal, dropout_p, softmax_scale):
    query = qkv[:, :, 0].permute(0, 2, 1, 3).contiguous()
    key = qkv[:, :, 1].permute(0, 2, 1, 3).contiguous()
    value = qkv[:, :, 2].permute(0, 2, 1, 3).contiguous()

    batch_size, nheads, seqlen, head_dim = query.shape
    valid_mask = _normalize_key_padding_mask(
        key_padding_mask,
        batch_size=batch_size,
        seqlen=seqlen,
        device=query.device,
    )
    atten_mask = (~valid_mask)[:, None, None, :].expand(batch_size, 1, seqlen, seqlen).contiguous()
    if causal:
        causal_mask = torch.triu(
            torch.ones((seqlen, seqlen), device=query.device, dtype=torch.bool),
            diagonal=1,
        )
        atten_mask = atten_mask | causal_mask.unsqueeze(0).unsqueeze(0)

    attn_out = torch_npu.npu_fusion_attention(
        query,
        key,
        value,
        head_num=nheads,
        input_layout="BNSD",
        atten_mask=atten_mask,
        scale=softmax_scale if softmax_scale is not None else head_dim**-0.5,
        keep_prob=1.0 - dropout_p,
    )[0]
    attn_out = attn_out.permute(0, 2, 1, 3).contiguous()
    attn_out = attn_out * valid_mask[:, :, None, None].to(attn_out.dtype)
    return attn_out


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
