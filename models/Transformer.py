
"""
完全解耦的Transformer模型用于船舶交通流多船轨迹预测
输入: [B, T, N, D] - (batch_size, time_frames, num_ships, features)
输出: [B, T, N, D] - 预测未来时间帧的轨迹

注意力层作为参数传入，完全解耦
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from layers.Transformer_Enc_Dec import Encoder, EncoderLayer, Decoder, DecoderLayer
from layers.Attention_family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding


# ==================== 完整模型 ====================

class Model(nn.Module):
    """
    船舶轨迹预测Transformer完整模型
    """
    def __init__(
        self,
        args
    ):
        super(Model, self).__init__()
        
        self.input_time_steps = args.seq_len
        self.output_time_steps = args.pred_len
        self.num_ships = args.num_ships
        self.num_features = args.num_features
        self.output_attention = args.output_attention

        
        # 输入特征维度 (N * D)
        self.enc_in = args.num_ships * args.num_features
        self.dec_in = args.num_ships * 2
        self.c_out = args.num_ships * 2
        
        # 编码器嵌入
        self.enc_embedding = DataEmbedding(self.enc_in, args.d_model, args.dropout)
        
        # 解码器嵌入
        self.dec_embedding = DataEmbedding(self.dec_in, args.d_model, args.dropout)
        
        # 编码器 - 注意力层作为参数传入
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, args.factor, attention_dropout=args.dropout, output_attention=args.output_attention),
                        args.d_model, args.n_heads
                    ),
                    args.d_model,
                    args.d_ff,
                    dropout=args.dropout,
                    activation=args.activation
                ) for _ in range(args.e_layers)
            ],
            conv_layers=None,
            norm_layer=nn.LayerNorm(args.d_model)
        )
        
        # 解码器 - 注意力层作为参数传入
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        FullAttention(True, args.factor, attention_dropout=args.dropout, output_attention=False),
                        args.d_model, args.n_heads
                    ),
                    AttentionLayer(
                        FullAttention(False, args.factor, attention_dropout=args.dropout, output_attention=False),
                        args.d_model, args.n_heads
                    ),
                    args.d_model,
                    args.d_ff,
                    dropout=args.dropout,
                    activation=args.activation
                ) for _ in range(args.d_layers)
            ],
            norm_layer=nn.LayerNorm(args.d_model),
            projection=nn.Linear(args.d_model, self.c_out, bias=True)
        )
    
    def forward(self, x_enc, x_dec=None, enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None, mask_x=None, mask_y=None):
        """
        前向传播
        
        Args:
            x_enc: [B, T_in, N, D] - 编码器输入（历史轨迹）
            x_dec: [B, T_out, N, D] - 解码器输入（目标轨迹，训练时使用）
            enc_self_mask: 编码器自注意力mask
            dec_self_mask: 解码器自注意力mask（causal mask）
            dec_enc_mask: 解码器-编码器交叉注意力mask
        
        Returns:
            [B, T_out, N, D] - 预测的未来轨迹
        """
        batch_size = x_enc.size(0)
        device = x_enc.device
        ###########################################################
        # 构造时间步的 key padding mask
        time_valid_enc = mask_x.any(dim=2)  # [B, T_in]
        enc_kpm = ~time_valid_enc

        time_valid_dec = mask_y.any(dim=2)
        dec_kpm = ~time_valid_dec
        ###########################################################

        # 1. Reshape输入: [B, T, N, D] -> [B, T, N*D]
        x_enc_reshaped = x_enc.view(batch_size, self.input_time_steps, -1)
        
        # 2. 编码器
        enc_out = self.enc_embedding(x_enc_reshaped)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask, src_key_padding_mask=enc_kpm)
        
        # 3. 解码器
        if x_dec is not None:
            # 训练模式：使用teacher forcing
            x_dec_reshaped = x_dec.view(batch_size, self.output_time_steps, -1)
            dec_out = self.dec_embedding(x_dec_reshaped)
            dec_out = self.decoder(
                dec_out, enc_out,
                x_mask=dec_self_mask, cross_mask=dec_enc_mask,
                tgt_key_padding_mask=dec_kpm,          # ← 新增
                memory_key_padding_mask=enc_kpm        # ← 新增
            )
        else:
            # 推理模式：自回归生成
            x_last_for_dec = x_enc[:, -1, :, :2].reshape(batch_size, 1, -1)  # [B,1,N*D_dec] == [B,1,dec_in]

            dec_out = self.autoregressive_decode(
                enc_out=enc_out,
                mask_x=mask_x,                   # 构造 memory_key_padding_mask
                start_token=x_last_for_dec,      # 与训练 teacher forcing 对齐
                device=x_enc.device
            )
        
        # 4. Reshape输出: [B, T_out, N*D] -> [B, T_out, N, D]
        dec_out = dec_out.view(batch_size, self.output_time_steps, self.num_ships, 2)
        dec_out = dec_out * mask_y.unsqueeze(-1).float()

        # print('dec_out:', dec_out[0, 0, -1, :2])

        return dec_out
    
    # def autoregressive_decode(self, enc_out, batch_size, device):
    #     """
    #     自回归解码（推理模式）
        
    #     Args:
    #         enc_out: [B, T_in, d_model]
    #         batch_size: batch size
    #         device: device
        
    #     Returns:
    #         [B, T_out, c_out]
    #     """
    #     # 初始化解码器输入（用零向量）
    #     dec_input = torch.zeros(batch_size, 1, self.dec_in, device=device)
    #     outputs = []
        
    #     for t in range(self.output_time_steps):
    #         # 嵌入
    #         dec_embedded = self.dec_embedding(dec_input)
            
    #         # 解码
    #         dec_out = self.decoder(dec_embedded, enc_out, x_mask=None, cross_mask=None)
            
    #         # 取最后一个时间步
    #         pred = dec_out[:, -1:, :]  # [B, 1, c_out]
    #         outputs.append(pred)
            
    #         # 更新解码器输入
    #         dec_input = torch.cat([dec_input, pred], dim=1)
        
    #     # 拼接所有输出
    #     output = torch.cat(outputs, dim=1)  # [B, T_out, c_out]
        
    #     return output


    def autoregressive_decode(self, enc_out, mask_x, start_token, device):
        """
        自回归解码（推理模式）
        Args:
            enc_out:   [B, T_in, d_model]  编码器输出
            mask_x:    [B, T_in, N]        encoder 的有效掩码（构造 memory_kpm）
            start_token: [B, 1, dec_in]    第一个解码步的输入（建议用 x_enc 的最后一帧展平到 dec_in）
            device:    torch.device
        Returns:
            [B, T_out, c_out]  （要求 c_out == dec_in）
        """
        B = enc_out.size(0)
        T_out = self.output_time_steps

        # 1) encoder 侧的 key padding mask（屏蔽空时间步）
        enc_kpm = ~mask_x.any(dim=2)  # [B, T_in]  True=屏蔽

        # 2) 准备因果 mask（上三角 True=不允许注意）
        causal = torch.triu(
            torch.ones(T_out, T_out, dtype=torch.bool, device=device),
            diagonal=1
        )

        # 3) 初始化解码器输入（用与训练一致的起始 token）
        # start_token 形状必须是 [B,1, self.dec_in]
        dec_input = start_token.to(device)  # [B, 1, dec_in]

        outputs = []
        for t in range(T_out):
            # 嵌入
            dec_emb = self.dec_embedding(dec_input)                # [B, L, d_model]

            # 目标侧 padding mask：当前步内都有效 → 全 False
            dec_kpm = torch.zeros(B, dec_emb.size(1), dtype=torch.bool, device=device)

            # 因果 mask 切片到当前长度 L
            L = dec_emb.size(1)
            dec_self_mask = causal[:L, :L]                         # [L,L]

            # 解码（带上两种 padding mask）
            dec_out = self.decoder(
                x=dec_emb,
                cross=enc_out,
                x_mask=dec_self_mask,              # 自注意力因果
                cross_mask=None,
                tgt_key_padding_mask=dec_kpm,      # 目标侧 KPM（当前 L 内无 padding）
                memory_key_padding_mask=enc_kpm    # 编码侧 KPM（屏蔽空时间步）
            )                                       # [B, L, c_out]  (你的 Decoder 内部已做 projection)

            # 取最后一个时间步的输出
            step = dec_out[:, -1:, :]              # [B, 1, c_out]

            outputs.append(step)

            # 回填到下一步输入（要求 c_out == dec_in）
            dec_input = torch.cat([dec_input, step], dim=1)   # [B, L+1, dec_in]

        out = torch.cat(outputs, dim=1)                        # [B, T_out, c_out]
        return out

# ==================== 示例使用 ====================
