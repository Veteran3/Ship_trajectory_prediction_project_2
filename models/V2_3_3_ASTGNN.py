import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
import numpy as np
from utils.get_loss_function import get_loss_function

"""
针对 V2.3.0 版本 ASTGNN 的改进版

版本优化：
Edge-GNN。
与DataLoader中的_edge_features函数配合使用，

edge-GNN融合两个矩阵：
edge-features: 边特征矩阵，大小为 [num_edges, num_features]
66.7%稀疏度的先验矩阵

"""

# 矩阵稀疏性测试
def analyze_sparsity(matrix, name="Matrix"):
    """
    计算并打印矩阵的稀疏度分析报告。
    支持 numpy.ndarray 和 torch.Tensor。
    """
    # 1. 统一转为 numpy
    if isinstance(matrix, torch.Tensor):
        matrix = matrix.detach().cpu().numpy()
    
    # 2. 基础统计
    total_elements = matrix.size
    non_zero_elements = np.count_nonzero(matrix)
    zero_elements = total_elements - non_zero_elements
    
    sparsity = zero_elements / total_elements
    density = non_zero_elements / total_elements
    
    print(f"--- {name} Sparsity Analysis ---")
    print(f"Shape: {matrix.shape}")
    print(f"Total Elements: {total_elements}")
    print(f"Non-Zero: {non_zero_elements} ({density*100:.4f}%)")
    print(f"Zero:     {zero_elements} ({sparsity*100:.4f}%)")
    
    # 3. 如果是多通道矩阵 [..., Channels]，分别分析每个通道
    #    假设最后一个维度是通道
    if matrix.ndim > 2 and matrix.shape[-1] <= 10: # 简单的启发式判断
        num_channels = matrix.shape[-1]
        print(f"\n--- Per-Channel Breakdown ({num_channels} channels) ---")
        
        channel_names = ["Head-on", "Starboard", "Port", "Overtaking", "Channel 4", "Channel 5"]
        
        for k in range(num_channels):
            # 取出第 k 个通道的所有数据
            channel_data = matrix[..., k]
            c_total = channel_data.size
            c_nz = np.count_nonzero(channel_data)
            c_sparsity = 1.0 - (c_nz / c_total)
            
            c_name = channel_names[k] if k < len(channel_names) else f"Channel {k}"
            print(f"  {c_name}: Sparsity = {c_sparsity*100:.4f}% | Non-zeros = {c_nz}")
    print("-" * 30 + "\n")


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
    Edge-conditioned Dynamic Spatial GNN
    同时使用：
      - edge_features: 交互物理特征 (例如 1/d, v_rel, cosθ, sinθ ...)
      - A_prior: 先验图 (比如你 66.7% 稀疏的融合矩阵)

    输入:
        x:             [B, N, T, D]         节点特征 (船舶状态序列)
        edge_features: [B*T, N, N, edge_dim]
        A_prior:       [B*T, N, N] 或 [B, T, N, N]    (可选)
        entity_padding_mask:
                       [B, N] 或 [B, T, N] (True 表示该船/该时刻是 padding/不存在)

    输出:
        out:           [B, N, T, D]
    """

    def __init__(
        self,
        d_model: int,
        num_nodes: int,
        edge_dim: int = 4,
        hidden_edge: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_nodes = num_nodes
        self.dropout = nn.Dropout(dropout)

        d_attn = d_model  # 注意力空间维度，简单起见 = d_model

        self.A_prior_fusion = nn.Linear(5, 1, bias=False)

        # 节点特征映射到注意力空间 (Q, K, V)
        self.W_q = nn.Linear(d_model, d_attn, bias=False)
        self.W_k = nn.Linear(d_model, d_attn, bias=False)
        self.W_v = nn.Linear(d_model, d_attn, bias=False)

        # 边特征 MLP: 输入 [q_i, k_j, edge_ij]，输出一个标量 logit_ij
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * d_attn + edge_dim, hidden_edge),
            nn.ReLU(),
            nn.Linear(hidden_edge, 1),  # 不加 Sigmoid，直接当 logits
        )

        # 输出投影回 d_model
        self.Theta = nn.Linear(d_attn, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)

        # 可学习权重：控制物理项 & 先验项在 logits 中的影响力
        self.phys_weight = nn.Parameter(torch.tensor(1.0))
        self.prior_weight = nn.Parameter(torch.tensor(1.0))

    def forward(
        self,
        x: torch.Tensor,
        edge_features: torch.Tensor,
        A_prior: torch.Tensor = None,
        entity_padding_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        x:             [B, N, T, D]
        edge_features: [B*T, N, N, edge_dim]
        A_prior:       [B*T, N, N] 或 [B, T, N, N] (可选)
        entity_padding_mask:
                       [B, N] 或 [B, T, N] (True 表示 padding)
        """
        B, N, T, D = x.shape

        # [B, N, T, D] -> [B, T, N, D] -> [B*T, N, D]
        x_btnd = x.permute(0, 2, 1, 3).contiguous()
        Z = x_btnd.view(B * T, N, D)  # 当前层节点特征 H

        # 1. 节点映射到注意力空间
        Q = self.W_q(Z)  # [B*T, N, d_attn]
        K = self.W_k(Z)  # [B*T, N, d_attn]
        V = self.W_v(Z)  # [B*T, N, d_attn]

        d_attn = Q.size(-1)

        # 2. 基于节点特征的内容打分 (QK^T)
        content_logits = torch.matmul(Q, K.transpose(1, 2)) / math.sqrt(d_attn)
        # [B*T, N, N]

        # 3. 基于 (h_i, h_j, edge_ij) 的物理打分 (edge-conditioned)
        #    Qi: [B*T, N, 1, d], Kj: [B*T, 1, N, d]
        Qi = Q.unsqueeze(2).expand(-1, -1, N, -1)  # [B*T, N, N, d_attn]
        Kj = K.unsqueeze(1).expand(-1, N, -1, -1)  # [B*T, N, N, d_attn]

        # edge_features: [B*T, N, N, edge_dim]
        edge_input = torch.cat([Qi, Kj, edge_features], dim=-1)
        # [B*T, N, N, 2*d_attn + edge_dim]

        phys_logits = self.edge_mlp(edge_input).squeeze(-1)  # [B*T, N, N]

        # 4. 合并内容打分和物理打分
        logits = content_logits + self.phys_weight * phys_logits  # [B*T, N, N]

        # 5. 如果有先验邻接 A_prior，把它当成 log-prior 加进去
        if A_prior is not None:
            A_prior = self.A_prior_fusion(A_prior).squeeze(-1)

            # 支持 [B*T, N, N] 或 [B, T, N, N]
            if A_prior.dim() == 4:           # [B, T, N, N] -> [B*T, N, N]
                A_prior = A_prior.contiguous().view(B * T, N, N)
            # 如果你这边是 5 通道 [B,T,N,N,5]，请在外面先融合成 1 通道再传进来
            A_prior = torch.nan_to_num(A_prior, nan=0.0, posinf=0.0, neginf=0.0)
            A_prior = torch.clamp(A_prior, min=0.0)   # 所有负值直接截成 0
            eps = 1e-6
            prior_logits = torch.log(A_prior + eps)  # [B*T, N, N]
            logits = logits + self.prior_weight * prior_logits

        # 6. 实体 padding / 时间存在 mask
        if entity_padding_mask is not None:
            if entity_padding_mask.dim() == 2:
                # [B, N] -> [B, T, N]
                mask_bt = entity_padding_mask.unsqueeze(1).expand(B, T, N)
            else:
                mask_bt = entity_padding_mask  # [B, T, N]

            mask_bt = mask_bt.contiguous().view(B * T, N)  # [B*T, N]

            mask_row = mask_bt.unsqueeze(2).expand(B * T, N, N)  # i 无效
            mask_col = mask_bt.unsqueeze(1).expand(B * T, N, N)  # j 无效
            invalid = mask_row | mask_col

            # 用大负数而不是 -inf，避免 softmax 出 NaN
            logits = logits.masked_fill(invalid, -1e9)

        # 7. softmax 得到注意力权重 α_ij
        alpha = F.softmax(logits, dim=-1)  # [B*T, N, N]

        # 8. 消息聚合
        spatial_features = torch.matmul(alpha, V)  # [B*T, N, d_attn]
        output_features = self.Theta(spatial_features)  # [B*T, N, D]

        # 9. 残差 + LayerNorm + Dropout
        output_features = output_features.view(B, T, N, D)
        Z_reshaped = Z.view(B, T, N, D)

        out = self.norm(Z_reshaped + output_features)  # [B, T, N, D]
        out = self.dropout(out)

        # 10. 还原回 [B, N, T, D]
        out = out.permute(0, 2, 1, 3).contiguous()  # [B, N, T, D]

        # 11. 最后把 padding 节点特征清零（可选，但推荐）
        if entity_padding_mask is not None:
            if entity_padding_mask.dim() == 2:
                # [B, N] -> [B, N, 1, 1]
                mask_bn = entity_padding_mask.unsqueeze(-1).unsqueeze(-1)
            else:
                # [B, T, N] -> [B, N, T, 1]
                mask_bn = entity_padding_mask.permute(0, 2, 1).unsqueeze(-1)

            if mask_bn.dtype != torch.bool:
                mask_bn = mask_bn.bool()

            out = out.masked_fill(mask_bn, 0.0)

        return out  # [B, N, T, D]

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

    def forward(self, x, temporal_mask, entity_mask, A_social_t=None, edge_features=None):
        # x: (B, N, T, D)
        # 1. 时间注意力 [cite: 276]
        B, N, T, D = x.shape
        C = A_social_t.shape[-1]
        E = edge_features.shape[-1]
        A_social_t = A_social_t.reshape(B*T, N, N, C)
        edge_features = edge_features.reshape(B*T, N, N, E)
        x = self.sublayer_temporal(x, lambda x: self.temporal_attn(
            x, x, x, 
            key_padding_mask=temporal_mask
        ))
        
        # 2. 空间 GNN [cite: 277]
        x = self.sublayer_spatial(x, lambda x: self.spatial_gnn(
            x, 
            A_prior=A_social_t,
            edge_features=edge_features,
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

    def forward(self, x, memory, temporal_mask_self, temporal_mask_cross, entity_mask, attn_mask_self, A_social_t=None, edge_features_t=None):
        # x: (B, N, T_out, D)
        # memory: (B, N, T_in, D)
        B, N, L, D = x.shape
        C = A_social_t.shape[-1]
        E = edge_features_t.shape[-1]
        A_social_t_expanded = A_social_t.unsqueeze(1)
        A_social_t_repeated = A_social_t_expanded.repeat(1, L, 1, 1, 1)
        A_social_t_for_GNN = A_social_t_repeated.reshape(B * L, N, N, C)
        
        edge_features_t_expanded = edge_features_t.unsqueeze(1)
        edge_features_t_repeated = edge_features_t_expanded.repeat(1, L, 1, 1, 1)
        edge_features_t_for_GNN = edge_features_t_repeated.reshape(B * L, N, N, E)

        # 1. 时间自注意力 (Causal) [cite: 358]
        x = self.sublayer_self_attn(x, lambda x: self.self_attn(
            x, x, x, 
            key_padding_mask=temporal_mask_self, 
            attn_mask=attn_mask_self
        ))
        
        # 2. 空间 GNN [cite: 357]
        x = self.sublayer_spatial_attn(x, lambda x: self.spatial_gnn(
            x, 
            A_prior=A_social_t_for_GNN,
            edge_features=edge_features_t_for_GNN,
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
# [在 ShipASTGNN_Model 类内部]

class Model(nn.Module):
    
    def __init__(self, args):
        super(Model, self).__init__()
        
        # ... (num_nodes, in_features, d_model, num_heads 等参数不变) ...
        self.num_nodes = args.num_ships
        self.in_features = 15        # 这个地方应该改成5。COG被划分成了sin/cos两部分
        self.out_features = 2 
        self.d_model = args.d_model
        self.num_heads = args.n_heads
        self.num_layers = args.e_layers
        self.dropout = args.dropout
        self.kernel_size = getattr(args, 'kernel_size', 3)
        self.pred_len = args.pred_len
        self.sampling_prob = getattr(args, 'sampling_prob', 0.7)
        # 移除了 self.static_feature_dim

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

        # -- 嵌入层 (已修正) --
        self.src_input_proj = nn.Linear(self.in_features, self.d_model)
        self.trg_input_proj = nn.Linear(self.out_features, self.d_model)
        
        self.pos_encoder = TemporalPositionalEncoding(self.d_model, self.dropout)
        
        # [关键修正]
        # 恢复使用 "可学习的" 节点嵌入, 它不需要 x_static
        self.node_encoder = SpatialPositionalEncoding(self.num_nodes, self.d_model, self.dropout)
        # 移除了 self.static_encoder

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

        self.output_proj = nn.Linear(self.d_model, self.out_features)
        
        # [关键修正]
        # 初始化 Loss 函数 (与之前相同)
        self.criterion = get_loss_function(args.loss)


    def _compute_truth_deltas(self, x_enc, y_truth_abs):
        """
        [内部辅助方法] (保持不变)
        """
        # ... (代码与上一条回复完全相同)
        last_known_pos = x_enc[:, -1:, :, :2] 
        prev_future_pos = y_truth_abs[:, :-1, :, :2]
        all_previous_positions = torch.cat([last_known_pos, prev_future_pos], dim=1)
        y_truth_deltas = y_truth_abs[..., :2] - all_previous_positions
        return y_truth_deltas

    def forward(self, x_enc, y_truth_abs, mask_x, mask_y, A_social_t=None, edge_features=None):
        """
        前向传播 (已升级为 V4 - "固定预定采样" + "稳定反馈")
        
        Args:
            x_enc (torch.Tensor): [B, T_in, N, D_in] - 绝对历史轨迹
            y_truth_abs (torch.Tensor): [B, T_out, N, 2] - 真实绝对坐标 (训练时) 或 None (推理时)
            mask_x (torch.Tensor): [B, T_in, N] - 历史掩码
            mask_y (torch.Tensor): [B, T_out, N] - 未来掩码
            x_static: (被忽略, 但保留以兼容 API)
        """
        device = x_enc.device
        
        # analyze_sparsity(A_social_t, "A_social_t")

        # 1. 准备 掩码 和 Encoder 输入
        mask_x_permuted = mask_x.permute(0, 2, 1) 
        mask_y_permuted = mask_y.permute(0, 2, 1) 
        entity_mask = mask_x.any(dim=1) 
        entity_padding_mask = ~entity_mask 
        temporal_padding_mask_enc = ~mask_x_permuted
        temporal_padding_mask_dec = ~mask_y_permuted
        attn_mask_self = subsequent_mask(self.pred_len, device)

        x_enc_permuted = x_enc.permute(0, 2, 1, 3) 
        enc_in = self.src_input_proj(x_enc_permuted)
        
        # [关键] 使用 "可学习的" 节点嵌入
        enc_in = self.node_encoder(enc_in) 
        enc_in = self.pos_encoder(enc_in)  

        # 2. Encoder (不变)
        memory = enc_in
        for layer in self.encoder_layers:
            memory = layer(
                memory, 
                temporal_mask=temporal_padding_mask_enc, 
                entity_mask=entity_padding_mask,
                A_social_t=A_social_t,
                edge_features=edge_features
            )
        memory = self.encoder_norm(memory)


        # 3. Decoder
        # [关键修改]
        # 我们现在统一了训练和推理逻辑：
        # - self.training = True: y_truth_abs 被用于 "预定采样"
        # - self.training = False: y_truth_abs 为 None, 强制 100% 自回归
        
        # 1. 起始 Token 是 "绝对坐标" (来自历史)
        y_input_abs = x_enc_permuted[:, :, -1:, :2] # [B, N, 1, 2]
        
        outputs_deltas = [] # 收集预测的增量
        
        # 准备 "真值" (用于采样)
        if self.training:
            # (B, T_out, N, 2) -> (B, N, T_out, 2)
            y_truth_abs_permuted = y_truth_abs.permute(0, 2, 1, 3) 

        # 2. 循环 T_out 步
        for t in range(self.pred_len):
            
            # 3. 嵌入 "绝对坐标" 序列
            dec_in = self.trg_input_proj(y_input_abs)
            L = dec_in.size(2)
            
            # [关键] 注入 "可学习的" 节点嵌入
            dec_in = self.node_encoder(dec_in)
            dec_in = self.pos_encoder(dec_in)
            
            # 4. 准备掩码
            temporal_mask_self = temporal_padding_mask_dec[:, :, :L]
            attn_mask_self_step = attn_mask_self[:, :L, :L]
            
            # 5. 运行 Decoder

            A_social_t_last = A_social_t[:, -1, :, :]
            edge_features_t = edge_features[:, -1, :, :]
            # print('A_social_t_last shape:', A_social_t_last.shape)

            dec_out = dec_in
            for layer in self.decoder_layers:
                dec_out = layer(
                    dec_out, memory,
                    temporal_mask_self=temporal_mask_self,
                    temporal_mask_cross=temporal_padding_mask_enc,
                    entity_mask=entity_padding_mask,
                    attn_mask_self=attn_mask_self_step, 
                    A_social_t=A_social_t_last,
                    edge_features_t=edge_features_t
                )
            dec_out = self.decoder_norm(dec_out)
            
            # 6. 预测 "增量" (只取最后一个)
            pred_step_delta = self.output_proj(dec_out[:, :, -1:, :]) # [B, N, 1, 2]
            outputs_deltas.append(pred_step_delta)
            
            # 7. [关键] "预定采样" 逻辑
            
            # (如果不在训练中) 或 (随机数 > 固定概率)
            # self.training 为 False 时, 100% 走 else
            use_truth = self.training and (torch.rand(1) < self.sampling_prob)
            
            if use_truth:
                # 使用 "真实" 绝对坐标
                # t=0 时, 拿到 T_out 的第 0 帧
                next_abs_pos = y_truth_abs_permuted[:, :, t:t+1, :] 
            else:
                # 使用 "预测" 的绝对坐标 (自回归)
                last_abs_pos = y_input_abs[:, :, -1:, :]
                next_abs_pos = last_abs_pos + pred_step_delta.detach() # 重建
            
            # 8. 将 "绝对坐标" 回填
            y_input_abs = torch.cat([y_input_abs, next_abs_pos], dim=2)

        # 9. 循环结束后, 收集所有 T_out 个 "预测的增量"
        output = torch.cat(outputs_deltas, dim=2) # [B, N, T_out, 2]
        
        
        # 4. Reshape (不变)
        output = output.permute(0, 2, 1, 3)
        final_mask = mask_y.unsqueeze(-1).float()
        
        return output * final_mask


    # def autoregressive_decode(self, memory, entity_padding_mask, temporal_padding_mask_enc, attn_mask_self, x_enc_permuted, x_static):
    #     """
    #     [已完善 - V3] 自回归解码
    #     输入: 绝对坐标
    #     输出: 增量
    #     内部循环: 绝对坐标
    #     """
    #     device = memory.device
        
    #     # 1. 起始 Token 是 "绝对坐标"
    #     y_input_abs = x_enc_permuted[:, :, -1:, :2] # [B, N, 1, 2]
        
    #     if x_static is not None:
    #          static_embed = self.static_encoder(x_static.to(device))
    #          static_embed_expanded = static_embed.unsqueeze(2)
        
    #     outputs_deltas = [] # 收集预测的增量
        
    #     for t in range(self.pred_len):
            
    #         # 2. 嵌入 "绝对坐标" 序列
    #         dec_in = self.trg_input_proj(y_input_abs)
    #         L = dec_in.size(2)
            
    #         if x_static is not None:
    #             dec_in = dec_in + static_embed_expanded.repeat(1, 1, L, 1)
    #         else:
    #             dec_in = self.node_encoder(dec_in)
                
    #         dec_in = self.pos_encoder(dec_in)
            
    #         # 3. 准备掩码
    #         temporal_mask_self = torch.zeros(
    #             memory.size(0), self.num_nodes, L, dtype=torch.bool, device=device)
    #         attn_mask_self_step = attn_mask_self[:, :L, :L]
            
    #         # 4. 运行 Decoder
    #         dec_out = dec_in
    #         for layer in self.decoder_layers:
    #                 dec_out = layer(
    #                     dec_out, memory,
    #                     temporal_mask_self=temporal_mask_self,
    #                     temporal_mask_cross=temporal_padding_mask_enc,
    #                     entity_mask=entity_padding_mask,
    #                     attn_mask_self=attn_mask_self_step
    #                 )
    #         dec_out = self.decoder_norm(dec_out)
            
    #         # 5. [关键] 预测 "增量" (只取最后一个)
    #         pred_step_delta = self.output_proj(dec_out[:, :, -1:, :]) # [B, N, 1, 2]
    #         outputs_deltas.append(pred_step_delta)
            
    #         # 6. [关键] 重建 "绝对坐标"
    #         last_abs_pos = y_input_abs[:, :, -1:, :]
    #         next_abs_pos = last_abs_pos + pred_step_delta # 计算 t+1 的绝对位置
            
    #         # 7. [关键] 将 "绝对坐标" 回填
    #         y_input_abs = torch.cat([y_input_abs, next_abs_pos.detach()], dim=2) 

    #     # 8. 拼接所有 "预测的增量"
    #     output = torch.cat(outputs_deltas, dim=2) # [B, N, T_out, 2]
    #     return output

# [在 ShipASTGNN_Model 类内部]

    def loss(self, pred_deltas, y_truth_abs, x_enc, mask_y):
        """
        封装的损失计算 (已升级)
        
        Args:
            pred_deltas (torch.Tensor): [B, T_out, N, 2] - 模型的 "增量" 输出
            y_truth_abs (torch.Tensor): [B, T_out, N, 2] - 真实的 "绝对" 坐标
            x_enc (torch.Tensor): [B, T_in, N, D_in] - 历史轨迹
            mask_y (torch.Tensor): [B, T_out, N] - 掩码 (True=有效)
        
        Returns:
            torch.Tensor: loss_delta (用于反向传播)
            torch.Tensor: loss_absolute (用于日志打印, .item() 获取)
        """
        y_truth_abs = y_truth_abs[..., :2]
        # 1. 准备掩码
        if mask_y.dtype == torch.float:
            mask_y_bool = mask_y.bool()
        else:
            mask_y_bool = mask_y

        # 2. 计算 "真实增量"
        y_truth_deltas = self._compute_truth_deltas(x_enc, y_truth_abs)
        
        # 3. 计算 "增量损失" (用于反向传播)
        #    只在有效点上计算
        loss_delta = self.criterion(
            pred_deltas[mask_y_bool], 
            y_truth_deltas[mask_y_bool]
        )
        
        # 4. 计算 "绝对损失" (用于日志)
        
        # 4.1 重建 "绝对预测"
        pred_absolute = self.integrate(pred_deltas, x_enc)
        
        # 4.2 计算 "绝对损失"
        #     只在有效点上计算
        loss_absolute = self.criterion(
            pred_absolute[mask_y_bool],
            y_truth_abs[mask_y_bool]
        )
        
        # 5. 返回两个值
        return loss_delta, loss_absolute

    def integrate(self, pred_deltas, x_enc_history):
        """
        [公共方法] (保持不变, 它不需要 x_static)
        """
        last_known_pos = x_enc_history[:, -1:, :, :2].to(pred_deltas.device)
        cumulative_deltas = torch.cumsum(pred_deltas, dim=1)
        outputs_absolute = last_known_pos + cumulative_deltas
        return outputs_absolute