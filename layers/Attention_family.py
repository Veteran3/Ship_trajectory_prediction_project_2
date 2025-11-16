import torch
from torch import nn
import math
import torch.nn.functional as F
# ==================== 注意力机制 ====================

class TrendAwareAttentionLayer(nn.Module):
    """
    ASTGNN 风格的注意力层
    
    核心思想：
    使用 1D 卷积 (nn.Conv1d) 来替代标准 Transformer 的
    全连接层 (nn.Linear) 来生成 Q 和 K。
    
    这使得模型在计算注意力之前，就能"感知"到时间序列的局部趋势。
    
    - 'causal=True' (用于 Decoder 自注意力): 
      使用因果卷积 (padding 在左侧)，确保 t 时刻的 Q/K 
      只能看到 (t, t-1, t-2...) 的信息。
    - 'causal=False' (用于 Encoder 和 Cross-Attention): 
      使用标准 1D 卷积 (padding='same')，
      t 时刻的 Q/K 可以看到 (t-1, t, t+1) 的信息。
    """
    def __init__(self, d_model, n_heads, kernel_size=3, causal=False, dropout=0.1):
        super(TrendAwareAttentionLayer, self).__init__()
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.causal = causal
        self.kernel_size = kernel_size
        
        # --- 关键修改 ---
        # 1. 因果卷积 (用于 Decoder self-attn)
        if self.causal:
            # 确保卷积核只看到 "过去"
            # padding = (kernel_size - 1) 在左侧
            self.causal_padding = (self.kernel_size - 1, 0) 
            self.conv_q = nn.Conv1d(d_model, d_model, self.kernel_size)
            self.conv_k = nn.Conv1d(d_model, d_model, self.kernel_size)
        
        # 2. 标准 1D 卷积 (用于 Encoder 和 Cross-attn)
        else:
            # padding='same' 保持长度不变
            # (kernel_size - 1) // 2 和 kernel_size // 2 
            # 是 PyTorch 中实现 'same' padding 的标准方法
            pad_left = (self.kernel_size - 1) // 2
            pad_right = self.kernel_size // 2
            self.same_padding = (pad_left, pad_right)
            self.conv_q = nn.Conv1d(d_model, d_model, self.kernel_size)
            self.conv_k = nn.Conv1d(d_model, d_model, self.kernel_size)
        # ----------------

        # V 保持不变，仍然是线性投影
        self.linear_v = nn.Linear(d_model, d_model)
        self.linear_out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, attn_mask=None, src_key_padding_mask=None, memory_key_padding_mask=None, tgt_key_padding_mask=None, tau=None, delta=None):
        """
        前向传播 (签名与您的 DecoderLayer 完美兼容)
        
        Args:
            queries: [B, T_q, D]
            keys:    [B, T_k, D]
            values:  [B, T_k, D]
            attn_mask (x_mask): [T_q, T_k] (因果掩码)
            src_key_padding_mask (enc_kpm): [B, T_k] (Encoder的KPM)
            memory_key_padding_mask (enc_kpm): [B, T_k] (Cross-Attn的KPM)
            tgt_key_padding_mask (dec_kpm_step): [B, T_q] (Decoder的KPM)
        """
        B, T_q, _ = queries.shape
        B, T_k, _ = keys.shape
        B, T_v, _ = values.shape # T_k == T_v

        # --- 1. Q, K 卷积投影 ---
        
        # (B, T, D) -> (B, D, T) 以便 1D 卷积
        q_conv_in = queries.transpose(1, 2)
        k_conv_in = keys.transpose(1, 2)

        # 应用 padding
        if self.causal:
            q_conv_in = F.pad(q_conv_in, self.causal_padding)
            k_conv_in = F.pad(k_conv_in, self.causal_padding)
        else:
            q_conv_in = F.pad(q_conv_in, self.same_padding)
            k_conv_in = F.pad(k_conv_in, self.same_padding)

        # 卷积
        # (B, D, T+pad) -> (B, D, T)
        q_conv_out = self.conv_q(q_conv_in)
        k_conv_out = self.conv_k(k_conv_in)
        
        # (B, D, T) -> (B, T, D)
        q_out = q_conv_out.transpose(1, 2)
        k_out = k_conv_out.transpose(1, 2)

        # --- 2. V 线性投影 ---
        v_out = self.linear_v(values)
        
        # --- 3. 拆分多头 ---
        # (B, T, D) -> (B, T, H, D_k) -> (B, H, T, D_k)
        Q = q_out.view(B, T_q, self.n_heads, self.d_k).transpose(1, 2)
        K = k_out.view(B, T_k, self.n_heads, self.d_k).transpose(1, 2)
        V = v_out.view(B, T_v, self.n_heads, self.d_k).transpose(1, 2)

        # --- 4. 注意力计算 ---
        # (B, H, T_q, D_k) @ (B, H, D_k, T_k) -> (B, H, T_q, T_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # --- 5. 应用掩码 ---
        # (兼容您的 Encoder 和 Decoder)
        
        # 5.1 键填充掩码 (Key Padding Mask)
        # (来自 Encoder, 或者 Cross-Attention)
        key_padding_mask = memory_key_padding_mask if memory_key_padding_mask is not None else src_key_padding_mask
        
        # (来自 Decoder Self-Attention)
        if key_padding_mask is None:
            key_padding_mask = tgt_key_padding_mask
            
        if key_padding_mask is not None:
            # (B, T_k) -> (B, 1, 1, T_k)
            scores = scores.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))
            
        # 5.2 注意力掩码 (Causal Mask)
        if attn_mask is not None:
            # (T_q, T_k) -> (1, 1, T_q, T_k)
            scores = scores.masked_fill(attn_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
            
        # --- 6. Softmax & Dropout ---
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # --- 7. 合并 V ---
        # (B, H, T_q, T_k) @ (B, H, T_k, D_k) -> (B, H, T_q, D_k)
        attn_output = torch.matmul(attn_weights, V)
        
        # --- 8. 合并多头 ---
        # (B, H, T_q, D_k) -> (B, T_q, H, D_k) -> (B, T_q, D)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T_q, self.d_model)
        
        return self.linear_out(attn_output), attn_weights

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