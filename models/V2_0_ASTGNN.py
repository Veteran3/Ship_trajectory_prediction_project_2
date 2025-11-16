import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
import numpy as np
from utils.get_loss_function import get_loss_function
# ----------------------------------------------------------------------
# 模块 1: 帮助函数 (来自您之前的文件)
# ----------------------------------------------------------------------
def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def subsequent_mask(size, device):
    """
    生成一个上三角矩阵的因果掩码。
    """
    attn_shape = (1, size, size)
    mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    # 必须 .to(device)
    return (torch.from_numpy(mask) == 0).to(device) # (1, T, T), True=允许, False=屏蔽

class SublayerConnection(nn.Module):
    """
    残差连接 + LayerNorm (来自 ASTGNN 论文 [cite: 271])
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        """
        x: (B, N, T, D)
        sublayer: 一个函数或 nn.Module
        """
        return x + self.dropout(sublayer(self.norm(x)))

# ----------------------------------------------------------------------
# 模块 2: 时间趋势感知注意力 (来自 ASTGNN [cite: 276, 310])
# ----------------------------------------------------------------------
class TrendAwareAttention(nn.Module):
    """
    ASTGNN 的核心：时间趋势感知注意力 (1D 卷积注意力)
    在 "T" 维度上操作
    """
    def __init__(self, d_model, num_heads, kernel_size=3, mode='1d', dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.mode = mode
        self.kernel_size = kernel_size

        if self.mode == 'causal':
            # 因果卷积 (Decoder) [cite: 359]
            self.causal_padding = (self.kernel_size - 1)
            self.conv_q = nn.Conv1d(d_model, d_model, kernel_size)
            self.conv_k = nn.Conv1d(d_model, d_model, kernel_size)
        else:
            # 标准 1D 卷积 (Encoder) [cite: 312]
            self.conv_q = nn.Conv1d(d_model, d_model, kernel_size, padding='same')
            self.conv_k = nn.Conv1d(d_model, d_model, kernel_size, padding='same')

        self.linear_v = nn.Linear(d_model, d_model)
        self.linear_out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, key_padding_mask=None, attn_mask=None):
        """
        输入:
        - query, key, value: (B, N, T, D)
        - key_padding_mask: (B, N, T) [True=无效]
        - attn_mask: (T, T) [False=无效]
        """
        B, N, T_q, D = query.shape
        _, _, T_k, _ = key.shape
        
        # 1. 准备 1D 卷积输入 (T 维度是序列维度)
        # (B, N, T, D) -> (B*N, T, D) -> (B*N, D, T)
        q_conv_in = query.reshape(B*N, T_q, D).transpose(1, 2)
        k_conv_in = key.reshape(B*N, T_k, D).transpose(1, 2)
        
        # 2. 应用 1D 卷积
        if self.mode == 'causal':
            q_conv_in = F.pad(q_conv_in, (self.causal_padding, 0))
            k_conv_in = F.pad(k_conv_in, (self.causal_padding, 0))
        
        q_conv_out = self.conv_q(q_conv_in).transpose(1, 2) # (B*N, T_q, D)
        k_conv_out = self.conv_k(k_conv_in).transpose(1, 2) # (B*N, T_k, D)
        
        # 3. Reshape 并计算 V
        # (B*N, T, D) -> (B, N, T, H, D_k) -> (B, N, H, T, D_k)
        Q = q_conv_out.reshape(B, N, T_q, self.num_heads, self.d_k).transpose(2, 3)
        K = k_conv_out.reshape(B, N, T_k, self.num_heads, self.d_k).transpose(2, 3)
        
        V = self.linear_v(value).view(B, N, T_k, self.num_heads, self.d_k).transpose(2, 3)

        # 4. 计算注意力 (B, N, H, T_q, T_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # 5. 应用掩码
        if key_padding_mask is not None:
            # (B, N, T_k) -> (B, N, 1, 1, T_k)
            mask = key_padding_mask.unsqueeze(2).unsqueeze(3)
            scores = scores.masked_fill(mask, -1e9)
            
        if attn_mask is not None:
            # (T_q, T_k) -> (1, 1, 1, T_q, T_k)
            scores = scores.masked_fill(attn_mask.unsqueeze(0).unsqueeze(0).unsqueeze(0) == 0, -1e9)

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # (B, N, H, T_q, D_k)
        x = torch.matmul(attn, V)
        
        # (B, N, T_q, D_model)
        x = x.transpose(2, 3).contiguous().view(B, N, T_q, self.d_model)

        return self.linear_out(x)

# ----------------------------------------------------------------------
# 模块 3: 动态空间 GNN (来自 ASTGNN [cite: 320, 332])
# ----------------------------------------------------------------------
class DynamicSpatialGNN(nn.Module):
    """
    ASTGNN 的核心：动态 GCN (spatialAttentionScaledGCN)
    在 "N" 维度上操作
    
    原始论文公式: X_t = σ((A ⊙ S_t) * Z_t * W) 
    - A = 静态邻接矩阵 [cite: 329]
    - S_t = 动态自注意力矩阵 
    
    我们的适配:
    由于没有静态 A, 我们假设 A 是全 1 矩阵 (全连接)
    公式简化为: X_t = σ((1 ⊙ S_t) * Z_t * W) = σ(S_t * Z_t * W)
    """
    def __init__(self, d_model, num_nodes, dropout=0.1):
        super(DynamicSpatialGNN, self).__init__()
        self.d_model = d_model
        self.num_nodes = num_nodes
        
        # 线性变换 W (来自 GCN [cite: 325, 348])
        self.Theta = nn.Linear(d_model, d_model, bias=False)
        
        # 注册一个全1的"静态邻接矩阵" A
        # 这是为了严格遵循论文 (A ⊙ S_t) 的逻辑 
        self.register_buffer('static_adj', torch.ones(num_nodes, num_nodes))
        
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, entity_padding_mask=None):
        """
        输入:
        - x: (B, N, T, D)
        - entity_padding_mask: (B, N) [True=无效]
        """
        B, N, T, D = x.shape
        
        # GNN/空间注意力 是在每个时间步 T 上独立计算的
        # (B, N, T, D) -> (B, T, N, D) -> (B*T, N, D)
        x_permuted = x.permute(0, 2, 1, 3).contiguous()
        Z = x_permuted.view(B*T, N, D)
        
        # 1. 计算 S_t: 动态注意力矩阵 
        # (B*T, N, D) @ (B*T, D, N) -> (B*T, N, N)
        S_t = torch.matmul(Z, Z.transpose(1, 2)) / math.sqrt(D) 
        
        # 2. 应用实体掩码 ("幽灵船")
        if entity_padding_mask is not None:
            # (B, N) -> (B, 1, N) -> (B*T, N)
            mask = entity_padding_mask.unsqueeze(1).repeat(1, T, 1)
            mask = mask.view(B*T, N)
            
            # (B*T, N) -> (B*T, N, 1) 和 (B*T, 1, N)
            # 屏蔽 S_t 矩阵中所有 "幽灵船" 相关的行和列
            mask_row = mask.unsqueeze(2)
            mask_col = mask.unsqueeze(1)
            S_t = S_t.masked_fill(mask_row, -1e9)
            S_t = S_t.masked_fill(mask_col, -1e9)
            
        S_t_softmax = F.softmax(S_t, dim=-1) # (B*T, N, N)
        
        # 3. 计算 GCN
        # A ⊙ S_t 
        # (N, N) ⊙ (B*T, N, N)
        adj_dynamic = self.static_adj.mul(S_t_softmax) 
        
        # (A ⊙ S_t) * Z_t 
        # (B*T, N, N) @ (B*T, N, D) -> (B*T, N, D)
        spatial_features = torch.matmul(adj_dynamic, Z)
        
        # (A ⊙ S_t) * Z_t * W 
        # (B*T, N, D) -> (B*T, N, D)
        output_features = F.relu(self.Theta(spatial_features))
        output_features = self.dropout(output_features)
        
        # 4. 恢复形状
        # (B*T, N, D) -> (B, T, N, D) -> (B, N, T, D)
        return output_features.view(B, T, N, D).permute(0, 2, 1, 3)

# ----------------------------------------------------------------------
# 模块 4: 嵌入层 (来自 ASTGNN [cite: 376, 387])
# ----------------------------------------------------------------------
class TemporalPositionalEncoding(nn.Module):
    """
    标准 Transformer 时间位置编码 [cite: 376]
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(TemporalPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).unsqueeze(0) # (1, 1, T_max, D)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (B, N, T, D)
        x = x + self.pe[:, :, :x.size(2), :]
        return self.dropout(x)

class SpatialPositionalEncoding(nn.Module):
    """
    可学习的节点嵌入 (可学习的船舶槽位嵌入)
    用于捕捉 "空间异质性" [cite: 387, 394]
    """
    def __init__(self, num_nodes, d_model, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(num_nodes, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        # x: (B, N, T, D)
        # self.embedding.weight: (N, D) -> (1, N, 1, D)
        # 自动广播
        return self.dropout(x + self.embedding.weight.unsqueeze(0).unsqueeze(2))

# ----------------------------------------------------------------------
# 模块 5: Encoder / Decoder 层 (来自 ASTGNN )
# ----------------------------------------------------------------------
class EncoderLayer(nn.Module):
    """
    ASTGNN Encoder Layer [cite: 268, 276]
    顺序: 1. 时间注意力 2. 空间 GNN
    """
    def __init__(self, d_model, temporal_attn, spatial_gnn, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.temporal_attn = temporal_attn
        self.spatial_gnn = spatial_gnn
        self.sublayer_temporal = SublayerConnection(d_model, dropout)
        self.sublayer_spatial = SublayerConnection(d_model, dropout)

    def forward(self, x, temporal_mask, entity_mask):
        # x: (B, N, T, D)
        # 1. 时间注意力 [cite: 276]
        x = self.sublayer_temporal(x, lambda x: self.temporal_attn(
            x, x, x, 
            key_padding_mask=temporal_mask
        ))
        
        # 2. 空间 GNN [cite: 277]
        x = self.sublayer_spatial(x, lambda x: self.spatial_gnn(
            x, 
            entity_padding_mask=entity_mask
        ))
        return x

class DecoderLayer(nn.Module):
    """
    ASTGNN Decoder Layer [cite: 268, 356]
    顺序: 1. 时间自注意力 2. 空间 GNN 3. 时间交叉注意力
    """
    def __init__(self, d_model, self_attn, cross_attn, spatial_gnn, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attn = self_attn
        self.cross_attn = cross_attn
        self.spatial_gnn = spatial_gnn
        self.sublayer_self_attn = SublayerConnection(d_model, dropout)
        self.sublayer_spatial_attn = SublayerConnection(d_model, dropout)
        self.sublayer_cross_attn = SublayerConnection(d_model, dropout)

    def forward(self, x, memory, temporal_mask_self, temporal_mask_cross, entity_mask, attn_mask_self):
        # x: (B, N, T_out, D)
        # memory: (B, N, T_in, D)
        
        # 1. 时间自注意力 (Causal) [cite: 358]
        x = self.sublayer_self_attn(x, lambda x: self.self_attn(
            x, x, x, 
            key_padding_mask=temporal_mask_self, 
            attn_mask=attn_mask_self
        ))
        
        # 2. 空间 GNN [cite: 357]
        x = self.sublayer_spatial_attn(x, lambda x: self.spatial_gnn(
            x, 
            entity_padding_mask=entity_mask
        ))
        
        # 3. 时间交叉注意力 [cite: 362]
        x = self.sublayer_cross_attn(x, lambda x: self.cross_attn(
            x, memory, memory, 
            key_padding_mask=temporal_mask_cross
        ))
        return x

# ----------------------------------------------------------------------
# 模块 6: 完整模型
# ----------------------------------------------------------------------
class Model(nn.Module):
    """
    忠实于 ASTGNN 论文逻辑的船舶轨迹预测模型
    """
    def __init__(self, args):
        super(Model, self).__init__()
        
        # 从 args 获取参数
        self.num_nodes = args.num_ships
        self.in_features = args.num_features
        self.out_features = 2 # 经纬度
        self.d_model = args.d_model
        self.num_heads = args.n_heads
        self.num_layers = args.e_layers # 假设 E 和 D 层数相同
        self.dropout = args.dropout
        self.kernel_size = getattr(args, 'kernel_size', 3)
        self.pred_len = args.pred_len
        self.sampling_prob = getattr(args, 'sampling_prob', 0.7) # 预定采样概率
        
        # -- 核心模块 --
        c = copy.deepcopy
        
        # 时间注意力 (1D Conv) [cite: 310]
        temporal_attn = TrendAwareAttention(
            self.d_model, self.num_heads, self.kernel_size, mode='1d', dropout=self.dropout)
        
        # 时间注意力 (Causal Conv) [cite: 359]
        temporal_attn_causal = TrendAwareAttention(
            self.d_model, self.num_heads, self.kernel_size, mode='causal', dropout=self.dropout)
        
        # 空间 GNN (Dynamic) [cite: 320, 332]
        spatial_gnn = DynamicSpatialGNN(
            self.d_model, self.num_nodes, dropout=self.dropout)

        # -- 嵌入层 --
        self.src_input_proj = nn.Linear(self.in_features, self.d_model)
        self.trg_input_proj = nn.Linear(self.out_features, self.d_model)
        
        self.pos_encoder = TemporalPositionalEncoding(self.d_model, self.dropout) 
        self.node_encoder = SpatialPositionalEncoding(self.num_nodes, self.d_model, self.dropout) 

        # -- Encoder -- [cite: 276]
        encoder_layer = EncoderLayer(
            self.d_model, c(temporal_attn), c(spatial_gnn), self.dropout)
        self.encoder_layers = clones(encoder_layer, self.num_layers)
        self.encoder_norm = nn.LayerNorm(self.d_model)

        # -- Decoder -- [cite: 356]
        decoder_layer = DecoderLayer(
            self.d_model, c(temporal_attn_causal), c(temporal_attn), c(spatial_gnn), self.dropout)
        self.decoder_layers = clones(decoder_layer, self.num_layers)
        self.decoder_norm = nn.LayerNorm(self.d_model)

        # -- 输出层 --
        self.output_proj = nn.Linear(self.d_model, self.out_features)

    def forward(self, x_enc, x_dec, mask_x, mask_y):
        """
        前向传播
        
        Args:
            x_enc: [B, T_in, N, D_in] - 编码器输入 (历史轨迹)
            x_dec: [B, T_out, N, 2]  - 解码器输入 (目标轨迹，训练时使用)
            mask_x: [B, T_in, N] - "实体掩码B" (历史)。True=有效船只。
            mask_y: [B, T_out, N] - "实体掩码B" (未来)。True=有效船只。
        """
        
        device = x_enc.device
        B, T_in, N, D_in = x_enc.shape
        # 1. 准备所有掩码 (这是您项目中最关键的部分)
        # (B, T, N) -> (B, N, T)
        mask_x_permuted = mask_x.permute(0, 2, 1) # (B, N, T_in)
        mask_y_permuted = mask_y.permute(0, 2, 1) # (B, N, T_out)
        
        # 1.1 实体掩码 (用于空间 GNN)
        # (B, N) - 只要在 T_in 出现过一次的船，就是有效实体
        entity_mask = mask_x.any(dim=1) # (B, N)
        entity_padding_mask = ~entity_mask # [True=无效]
        
        # 1.2 时间掩码 (用于时间注意力)
        # (B, N, T) - [True=无效]
        temporal_padding_mask_enc = ~mask_x_permuted
        
        # (B, N, T) - [True=无效]
        temporal_padding_mask_dec = ~mask_y_permuted
        
        # 1.3 因果掩码 (用于 Decoder 自注意力)
        # (T_out, T_out) [False=无效]
        attn_mask_self = subsequent_mask(self.pred_len, device)

        # 2. Encoder [cite: 276]
        # (B, T_in, N, D_in) -> (B, N, T_in, D_in)
        x_enc_permuted = x_enc.permute(0, 2, 1, 3) 
        
        enc_in = self.src_input_proj(x_enc_permuted)
        enc_in = self.node_encoder(enc_in) # 添加 (1, N, 1, D)
        enc_in = self.pos_encoder(enc_in)  # 添加 (1, 1, T_in, D)
        
        memory = enc_in
        for layer in self.encoder_layers:
            memory = layer(
                memory, 
                temporal_mask=temporal_padding_mask_enc, 
                entity_mask=entity_padding_mask
            )
        memory = self.encoder_norm(memory) # (B, N, T_in, D_model)

        # 3. Decoder
        # 训练和推理使用不同逻辑
        if self.training:
            # ========================
            # 训练: 预定采样
            # ========================
            
            # (B, T_out, N, 2) -> (B, N, T_out, 2)
            y_truth = x_dec.permute(0, 2, 1, 3)
            
            # 构造起始 Token (x_enc 的最后一帧)
            # (B, N, 1, 2)
            y_input = x_enc_permuted[:, :, -1:, :2]
            
            outputs = []
            for t in range(self.pred_len):
                # (B, N, L, 2) (L=t+1)
                
                # 嵌入当前序列
                dec_in = self.trg_input_proj(y_input)
                dec_in = self.node_encoder(dec_in)
                dec_in = self.pos_encoder(dec_in) # (B, N, L, D_model)
                
                L = dec_in.size(2)
                
                # 准备当前步的掩码
                temporal_mask_self = temporal_padding_mask_dec[:, :, :L]
                attn_mask_self_step = attn_mask_self[:, :L, :L]
                
                dec_out = dec_in
                for layer in self.decoder_layers:
                    dec_out = layer(
                        dec_out, memory,
                        temporal_mask_self=temporal_mask_self,
                        temporal_mask_cross=temporal_padding_mask_enc,
                        entity_mask=entity_padding_mask,
                        attn_mask_self=attn_mask_self_step
                    )
                dec_out = self.decoder_norm(dec_out) # (B, N, L, D_model)
                
                # 取最后一个时间步的预测
                pred_step = self.output_proj(dec_out[:, :, -1:, :]) # (B, N, 1, 2)
                outputs.append(pred_step)
                
                # 预定采样
                if t < self.pred_len - 1:
                    use_truth = torch.rand(1) < self.sampling_prob
                    if use_truth:
                        next_input_token = y_truth[:, :, t+1:t+2, :]
                    else:
                        next_input_token = pred_step.detach()
                    
                    y_input = torch.cat([y_input, next_input_token], dim=2)

            # 拼接 T_out 步的预测
            output = torch.cat(outputs, dim=2) # (B, N, T_out, 2)
            
        else:
            # ========================
            # 推理: 自回归
            # ========================
            y_input = x_enc_permuted[:, :, -1:, :2] # (B, N, 1, 2)
            
            for t in range(self.pred_len):
                dec_in = self.trg_input_proj(y_input)
                dec_in = self.node_encoder(dec_in)
                dec_in = self.pos_encoder(dec_in)
                
                L = dec_in.size(2)
                temporal_mask_self = torch.zeros(
                    B, N, L, dtype=torch.bool, device=device) # [True=无效]
                attn_mask_self_step = attn_mask_self[:, :L, :L]
                
                dec_out = dec_in
                for layer in self.decoder_layers:
                    dec_out = layer(
                        dec_out, memory,
                        temporal_mask_self=temporal_mask_self,
                        temporal_mask_cross=temporal_padding_mask_enc,
                        entity_mask=entity_padding_mask,
                        attn_mask_self=attn_mask_self_step
                    )
                dec_out = self.decoder_norm(dec_out)
                
                pred_step = self.output_proj(dec_out[:, :, -1:, :]) # (B, N, 1, 2)
                
                # 将预测结果回填
                y_input = torch.cat([y_input, pred_step], dim=2)

            output = y_input[:, :, 1:, :] # (B, N, T_out, 2) (去掉起始 token)
        
        # 4. Reshape 输出 & 最终掩码
        # (B, N, T_out, 2) -> (B, T_out, N, 2)
        output = output.permute(0, 2, 1, 3)
        
        # 应用您最终的 "幽灵船" 清零掩码
        # (B, T_out, N) -> (B, T_out, N, 1)
        final_mask = mask_y.unsqueeze(-1).float()
        
        return output * final_mask
    
    def get_loss(self, loss_name):
        """
        获取损失函数
        """
        criterion = get_loss_function(loss_name)
        return criterion