
"""
完全解耦的Transformer模型用于船舶交通流多船轨迹预测
输入: [B, T, N, D] - (batch_size, time_frames, num_ships, features)
输出: [B, T, N, D] - 预测未来时间帧的轨迹

注意力层作为参数传入，完全解耦
"""


"""
船舶交通流多船轨迹预测 Transformer 模型 (Encoder-Decoder)

=======================================================================
核心设计 (基于您的项目  和代码)：
=======================================================================

1.  **架构: [B, T, N, D] -> [B, T, N*D] (展平策略)**
    * 本模型将 N 艘船及其 D 个特征在 *同一时刻* 展平 (Flatten) 为一个单一的特征向量 (N*D)。
    * 输入形状: `[B, T_in, N, D_in]` -> `[B, T_in, N*D_in]`。
    * 输出形状: `[B, T_out, N, 2]`  <- `[B, T_out, N*2]`。

2.  **注意力机制: 纯时间 (Temporal) 注意力**
    * 由于 N 维度被展平，Transformer 的自注意力 (Self-Attention) *只* 在 `T` (时间) 维度上运作。
    * 它学习的是 "包含 N 艘船的整个系统" 随时间的演化规律。
    * 它 *不会* 显式地学习 T1 时刻船 A 和船 B 之间的空间交互。

3.  **双重掩码 (Dual Masking) 策略 (关键!)**
    * 为了解决 "注意力污染"  和 "0 值干扰训练" ，本模型使用了两种*不同*的掩码机制：

    * **掩码A: "时间步掩码" (Temporal Padding Mask)**
        * **问题**: 某些时间帧 (Time Frame) 可能一艘船都没有 (空帧)，是无效的。
        * **形状**: `[B, T]`
        * **实现**: `mask_x.any(dim=2)`。
        * **作用**: 在 `Encoder` 和 `Decoder` 中作为 `key_padding_mask` (KPM) 传入。
        * **目的**: 告诉注意力机制 "忽略这些完全无效的时间步"。

    * **掩码B: "实体掩码" (Entity Padding Mask)**
        * **问题**: 船舶数 < N (例如 < 17)，使用 0 填充 (0-Padding) 的 "幽灵船"。
        * **形状**: `[B, T, N]`
        * **实现**: *不在* 注意力层，而是在训练的其他部分。
        * **作用**:
            1.  **归一化**: 预处理时仅在 `mask==1` 上统计均值/方差。
            2.  **损失计算**: `loss = F.mse_loss(pred[mask], y[mask])` [cite: 35]。
            3.  **模型输出**: `forward` 函数最后一行 `dec_out * mask_y.unsqueeze(-1)`，将幽灵船的预测强制清零。

4.  **解码策略: Teacher Forcing vs. 自回归 (Autoregressive)**
    * `forward` 函数通过 `x_dec` 是否为 `None` 来区分训练和推理。
    * **训练 (Teacher Forcing)**: 使用 `x_dec` (真实目标)，导致 "分布偏移" (Exposure Bias)，可能使 MSE Loss 虚低，但推理效果差 (如您展示的轨迹图)。
    * **推理 (Autoregressive)**: `autoregressive_decode` 函数实现自回归，将上一步的预测作为下一步的输入，这与真实推理一致。

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from layers.Transformer_Enc_Dec import Encoder, EncoderLayer, Decoder, DecoderLayer
from layers.Attention_family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding

from utils.get_loss_function import get_loss_function

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
            x_enc: [B, T_in, N, D_in] - 编码器输入 (历史轨迹)
            x_dec: [B, T_out, N, 2]  - 解码器输入 (目标轨迹，训练时使用 Teacher Forcing)
            enc_self_mask: 编码器自注意力mask (通常为None)
            dec_self_mask: 解码器自注意力mask (因果 mask, 在 Decoder 内部自动生成或外部传入)
            dec_enc_mask: 交叉注意力mask (通常为None)
            mask_x: [B, T_in, N] - "实体掩码B" (历史)。True=有效船只。
            mask_y: [B, T_out, N] - "实体掩码B" (未来)。True=有效船只。

        Returns:
            [B, T_out, N, 2] - 预测的未来轨迹
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
        自回归解码（推理模式）[cite: 35]
        在 T_out 步上循环，每次将上一步的输出作为下一步的输入。
        这模拟了真实推理场景，用于解决 Teacher Forcing 导致的 "分布偏移" 问题。

        Args:
            enc_out:     [B, T_in, d_model]  编码器输出 (Memory)
            mask_x:      [B, T_in, N]        "实体掩码B"，用于构造 "掩码A"
            start_token: [B, 1, dec_in]    第一个解码步的输入 (即 x_enc 的最后一帧)
            device:      torch.device

        Returns:
            [B, T_out, c_out]  (c_out == dec_in == N*2)
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

    def get_loss(self, loss_name):
        """
        获取损失函数
        """
        criterion = get_loss_function(loss_name)
        return criterion



# ==================== 示例使用 ====================
