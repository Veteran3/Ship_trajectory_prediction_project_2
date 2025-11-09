import torch
from torch import nn
import math

# ==================== 注意力机制 ====================

class FullAttention(nn.Module):
    """
    完整的缩放点积注意力
    """
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.mask_flag = mask_flag
        self.factor = factor
        self.scale = scale
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
    
    def forward(self, queries, keys, values, attn_mask=None, tau=None, delta=None, src_key_padding_mask=None):
        """
        Args:
            queries: [B, L_q, H, D]
            keys: [B, L_k, H, D]
            values: [B, L_v, H, D]
            attn_mask: mask
            tau: 时间尺度参数（可选）
            delta: 时间差参数（可选）
        Returns:
            out: [B, L_q, H, D]
            attn: [B, H, L_q, L_k] or None
        """
        B, L_q, H, D = queries.shape
        _, L_k, _, _ = keys.shape
        
        scale = self.scale or 1. / math.sqrt(D)
        
        # 计算注意力分数
        scores = torch.einsum("blhd,bshd->bhls", queries, keys)
        
        if src_key_padding_mask is not None:
            # src_key_padding_mask: [B, T_k]  True=需要屏蔽
            mask = src_key_padding_mask[:, None, None, :]       # [B,1,1,T_k]
            scores = scores.masked_fill(mask, float('-inf'))

        # 应用mask
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = self._get_causal_mask(L_q, L_k, queries.device)
            
            scores.masked_fill_(attn_mask == 0, -1e9)
        
        # 注意力权重
        attn = self.dropout(torch.softmax(scale * scores, dim=-1))
        
        # 加权求和
        out = torch.einsum("bhls,bshd->blhd", attn, values)
        
        if self.output_attention:
            return out, attn
        else:
            return out, None
    
    def _get_causal_mask(self, L_q, L_k, device):
        """生成因果mask"""
        mask = torch.tril(torch.ones(L_q, L_k, device=device)).bool()
        return mask.unsqueeze(0).unsqueeze(0)  # [1, 1, L_q, L_k]


class AttentionLayer(nn.Module):
    """
    注意力层包装器
    包含多头注意力的线性投影
    """
    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None):
        super(AttentionLayer, self).__init__()
        
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)
        
        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask=None, tau=None, delta=None, src_key_padding_mask=None):
        """
        Args:
            queries: [B, L_q, d_model]
            keys: [B, L_k, d_model]
            values: [B, L_v, d_model]
            attn_mask: mask
            tau: 时间尺度参数
            delta: 时间差参数
        Returns:
            [B, L_q, d_model], attention_weights
        """
        B, L_q, _ = queries.shape
        _, L_k, _ = keys.shape
        _, L_v, _ = values.shape
        H = self.n_heads
        
        # 线性投影并分头
        queries = self.query_projection(queries).view(B, L_q, H, -1)
        keys = self.key_projection(keys).view(B, L_k, H, -1)
        values = self.value_projection(values).view(B, L_v, H, -1)
        
        # 注意力计算
        out, attn = self.inner_attention(queries, keys, values, attn_mask, tau=tau, delta=delta, src_key_padding_mask=src_key_padding_mask)
        
        # 合并多头
        out = out.contiguous().view(B, L_q, -1)
        
        # 输出投影
        return self.out_projection(out), attn