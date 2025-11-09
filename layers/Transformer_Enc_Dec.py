import torch
from torch import nn
from torch.nn import functional as F


# ==================== 编码器 ====================

class EncoderLayer(nn.Module):
    """
    Transformer编码器层
    注意力层作为参数传入
    """
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu
    
    def forward(self, x, attn_mask=None, tau=None, delta=None, src_key_padding_mask=None):
        """
        Args:
            x: [B, L, d_model]
            attn_mask: attention mask
            tau: 时间尺度参数
            delta: 时间差参数
        Returns:
            [B, L, d_model], attention_weights
        """
        # 自注意力
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask,
            tau=tau, delta=delta,
            src_key_padding_mask=src_key_padding_mask
        )
        x = x + self.dropout(new_x)
        
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        
        return self.norm2(x + y), attn


class Encoder(nn.Module):
    """
    Transformer编码器（多层堆叠）
    """
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None, tau=None, delta=None, src_key_padding_mask=None):
        """
        Args:
            x: [B, L, d_model]
            attn_mask: attention mask
            tau: 时间尺度参数
            delta: 时间差参数
        Returns:
            [B, L, d_model], list of attention_weights
        """
        attns = []
        if self.conv_layers is not None:
            for i, (attn_layer, conv_layer) in enumerate(zip(self.attn_layers, self.conv_layers)):
                delta = delta if i == 0 else None
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta, src_key_padding_mask=src_key_padding_mask)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x, tau=tau, delta=None, src_key_padding_mask=src_key_padding_mask)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta, src_key_padding_mask=src_key_padding_mask)
                attns.append(attn)
        
        if self.norm is not None:
            x = self.norm(x)
        
        return x, attns


# ==================== 解码器 ====================
class DecoderLayer(nn.Module):
    """
    Transformer解码器层
    自注意力和交叉注意力作为参数传入
    """
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu
    
    def forward(
        self,
        x,                     # [B, L_q, d_model] - 解码器输入
        cross,                 # [B, L_k, d_model] - 编码器输出
        x_mask=None,           # 自注意力 attn_mask（如因果）
        cross_mask=None,       # 交叉注意力 attn_mask（一般用不到）
        tau=None,
        delta=None,
        tgt_key_padding_mask=None,     # 新增：Decoder 自注意力的 KPM，形状 [B, L_q]，True=屏蔽
        memory_key_padding_mask=None,  # 新增：交叉注意力的 KPM，形状 [B, L_k]，True=屏蔽
    ):
        """
        Returns:
            [B, L_q, d_model]
        """
        # 自注意力（带因果/结构mask + 自身的KPM）
        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask,
            tau=tau, delta=None,
            # 这里的参数名与你 self_attention 的实现保持一致；
            # 如果实现里叫 src_key_padding_mask，就按这个名字传
            src_key_padding_mask=tgt_key_padding_mask
        )[0])
        x = self.norm1(x)
        
        # 交叉注意力（cross-attn），对 encoder memory 使用其 KPM
        x = x + self.dropout(self.cross_attention(
            x, cross, cross,
            attn_mask=cross_mask,
            tau=tau, delta=delta,
            src_key_padding_mask=memory_key_padding_mask
        )[0])
        
        y = x = self.norm2(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        
        return self.norm3(x + y)


class Decoder(nn.Module):
    """
    Transformer解码器（多层堆叠）
    """
    def __init__(self, layers, norm_layer=None, projection=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection
    
    def forward(
        self,
        x,                           # [B, L_q, d_model]
        cross,                       # [B, L_k, d_model]
        x_mask=None,                 # 自注意力 attn_mask（如因果）
        cross_mask=None,             # 交叉注意力 attn_mask
        tau=None,
        delta=None,
        tgt_key_padding_mask=None,     # 新增：Decoder 自注意力 KPM [B, L_q] True=屏蔽
        memory_key_padding_mask=None,  # 新增：交叉注意力 KPM [B, L_k] True=屏蔽
    ):
        """
        Returns:
            [B, L_q, c_out or d_model]
        """
        for layer in self.layers:
            x = layer(
                x, cross,
                x_mask=x_mask, cross_mask=cross_mask,
                tau=tau, delta=delta,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask
            )
        
        if self.norm is not None:
            x = self.norm(x)
        
        if self.projection is not None:
            x = self.projection(x)
        
        return x
