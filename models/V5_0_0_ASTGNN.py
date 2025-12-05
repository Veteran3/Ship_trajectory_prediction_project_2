import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
import numpy as np
from utils.get_loss_function import get_loss_function

"""
v4.0.0 的改进

v4中验证了 Encoder only 模型对于预测的有效性。
在v5版本中，进一步优化模型。

主要思路：结合PINN，捕捉船舶运行学行为。
"""

# ----------------------------------------------------------------------
# 模块 1: 帮助函数
# ----------------------------------------------------------------------
def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class SublayerConnection(nn.Module):
    """
    残差连接 + LayerNorm
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

# ----------------------------------------------------------------------
# 模块 2: 动力学趋势注意力 (只保留 Encoder 模式)
# ----------------------------------------------------------------------
class KinematicsTrendAttention(nn.Module):
    """
    动力学趋势注意力 (Kinematics-Trend Attention) - 修正版
    
    1. 物理感知: Q/K 由 (位置+速度+加速度) 生成，而非普通卷积。
    2. 幽灵船屏蔽: 严格保留 key_padding_mask 逻辑，防止填充数据干扰注意力。
    """
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # ----------------------------------------------------------------
        # 1. 动力学算子分支 (替代原有的 conv_q, conv_k)
        # ----------------------------------------------------------------
        
        # 分支 A: 基础状态 (Identity / 平滑)
        self.branch_pos = nn.Conv1d(d_model, d_model, kernel_size=3, padding='same', groups=d_model)
        
        # 分支 B: 速度感知 (Velocity / 一阶微分) - bias=False
        self.branch_vel = nn.Conv1d(d_model, d_model, kernel_size=3, padding='same', groups=d_model, bias=False)
        
        # 分支 C: 加速度感知 (Acceleration / 二阶微分) - bias=False
        self.branch_acc = nn.Conv1d(d_model, d_model, kernel_size=3, padding='same', groups=d_model, bias=False)
        
        # 初始化物理权重 (差分算子)
        self._init_physics_weights()
        
        # 融合层: 将 (Pos + Vel + Acc) 投影生成 Q 和 K
        # 输入 3*d_model -> 输出 d_model
        self.fusion_q = nn.Linear(3 * d_model, d_model)
        self.fusion_k = nn.Linear(3 * d_model, d_model)

        # ----------------------------------------------------------------
        # 2. 标准 Attention 组件 (V 和 Output)
        # ----------------------------------------------------------------
        self.linear_v = nn.Linear(d_model, d_model)
        self.linear_out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def _init_physics_weights(self):
        """初始化卷积核为物理微分算子"""
        with torch.no_grad():
            # 初始化速度核 [-1, 0, 1]
            self.branch_vel.weight.fill_(0)
            center = 3 // 2
            for i in range(self.branch_vel.in_channels):
                self.branch_vel.weight[i, 0, center-1] = -1.0
                self.branch_vel.weight[i, 0, center+1] = 1.0
            
            # 初始化加速度核 [1, -2, 1]
            self.branch_acc.weight.fill_(0)
            for i in range(self.branch_acc.in_channels):
                self.branch_acc.weight[i, 0, center-1] = 1.0
                self.branch_acc.weight[i, 0, center] = -2.0
                self.branch_acc.weight[i, 0, center+1] = 1.0

    def forward(self, query, key, value, key_padding_mask=None):
        """
        query, key, value: [B, N, T, D] (通常 encoder 中 q=k=v=x)
        key_padding_mask: [B, N, T] (True 表示是幽灵船/Padding, 需要被屏蔽)
        """
        B, N, T_q, D = query.shape
        _, _, T_k, _ = key.shape
        
        # ------------------------------------------------
        # Step 1: 提取动力学特征 (Physics Extraction)
        # 代替原来的 q_conv_in 和 k_conv_in
        # ------------------------------------------------
        # 准备 Conv 输入: [B*N, T, D] -> [B*N, D, T]
        # 注意：这里我们只对 Query 和 Key 做物理提取，用来计算相似度
        # Value (V) 通常保留原始语义
        
        # 这里假设 Self-Attention，query=key=x。
        # 为了代码复用，我们先对 query (即 x) 提取特征
        q_in = query.reshape(B*N, T_q, D).transpose(1, 2)
        k_in = key.reshape(B*N, T_k, D).transpose(1, 2)
        
        # 提取 Query 的物理特征
        q_pos = self.branch_pos(q_in).transpose(1, 2)
        q_vel = self.branch_vel(q_in).transpose(1, 2)
        q_acc = self.branch_acc(q_in).transpose(1, 2)
        q_combined = torch.cat([q_pos, q_vel, q_acc], dim=-1) # [BN, Tq, 3D]
        
        # 提取 Key 的物理特征
        k_pos = self.branch_pos(k_in).transpose(1, 2)
        k_vel = self.branch_vel(k_in).transpose(1, 2)
        k_acc = self.branch_acc(k_in).transpose(1, 2)
        k_combined = torch.cat([k_pos, k_vel, k_acc], dim=-1) # [BN, Tk, 3D]
        
        # ------------------------------------------------
        # Step 2: 生成 Q, K, V
        # ------------------------------------------------
        # Q, K 利用物理特征生成
        Q = self.fusion_q(q_combined).view(B, N, T_q, self.num_heads, self.d_k).transpose(2, 3)
        K = self.fusion_k(k_combined).view(B, N, T_k, self.num_heads, self.d_k).transpose(2, 3)
        
        # V 保持原始语义 (通过 Linear 生成)
        V = self.linear_v(value).view(B, N, T_k, self.num_heads, self.d_k).transpose(2, 3)

        # ------------------------------------------------
        # Step 3: Attention 计算 (带 Mask)
        # ------------------------------------------------
        # (B, N, H, T_q, T_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # 【关键修正】: 恢复你原本的屏蔽逻辑
        if key_padding_mask is not None:
            # key_padding_mask: [B, N, T_k]
            # 扩展为: [B, N, 1, 1, T_k] 以广播到 scores [B, N, H, T_q, T_k]
            mask = key_padding_mask.unsqueeze(2).unsqueeze(3)
            
            # 使用 -1e9 进行屏蔽，Softmax 后变为 0
            scores = scores.masked_fill(mask, -1e9)
            
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        context = torch.matmul(attn, V) # [B, N, H, T_q, D_k]
        
        # Restore shape
        context = context.transpose(2, 3).contiguous().view(B, N, T_q, self.d_model)

        return self.linear_out(context)

# ----------------------------------------------------------------------
# 模块 3: 动态空间 GNN (保持不变)
# ----------------------------------------------------------------------
class DynamicSpatialGNN(nn.Module):
    def __init__(self, d_model, num_nodes, edge_dim=4, hidden_edge=32, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_nodes = num_nodes
        self.dropout = nn.Dropout(dropout)

        d_attn = d_model
        self.A_prior_fusion = nn.Linear(5, 1, bias=False)
        self.W_q = nn.Linear(d_model, d_attn, bias=False)
        self.W_k = nn.Linear(d_model, d_attn, bias=False)
        self.W_v = nn.Linear(d_model, d_attn, bias=False)

        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * d_attn + edge_dim, hidden_edge),
            nn.ReLU(),
            nn.Linear(hidden_edge, 1),
        )

        self.Theta = nn.Linear(d_attn, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)
        self.phys_weight = nn.Parameter(torch.tensor(1.0))
        self.prior_weight = nn.Parameter(torch.tensor(1.0))

    def forward(self, x, edge_features, A_prior=None, entity_padding_mask=None):
        B, N, T, D = x.shape
        x_btnd = x.permute(0, 2, 1, 3).contiguous()
        Z = x_btnd.view(B * T, N, D)

        Q = self.W_q(Z)
        K = self.W_k(Z)
        V = self.W_v(Z)
        d_attn = Q.size(-1)

        content_logits = torch.matmul(Q, K.transpose(1, 2)) / math.sqrt(d_attn)

        Qi = Q.unsqueeze(2).expand(-1, -1, N, -1)
        Kj = K.unsqueeze(1).expand(-1, N, -1, -1)
        edge_input = torch.cat([Qi, Kj, edge_features], dim=-1)
        phys_logits = self.edge_mlp(edge_input).squeeze(-1)

        logits = content_logits + self.phys_weight * phys_logits

        if A_prior is not None:
            A_prior = self.A_prior_fusion(A_prior).squeeze(-1)
            if A_prior.dim() == 4:
                A_prior = A_prior.contiguous().view(B * T, N, N)
            A_prior = torch.nan_to_num(A_prior, nan=0.0, posinf=0.0, neginf=0.0)
            A_prior = torch.clamp(A_prior, min=0.0)
            eps = 1e-6
            prior_logits = torch.log(A_prior + eps)
            logits = logits + self.prior_weight * prior_logits

        if entity_padding_mask is not None:
            if entity_padding_mask.dim() == 2:
                mask_bt = entity_padding_mask.unsqueeze(1).expand(B, T, N)
            else:
                mask_bt = entity_padding_mask
            mask_bt = mask_bt.contiguous().view(B * T, N)
            mask_row = mask_bt.unsqueeze(2).expand(B * T, N, N)
            mask_col = mask_bt.unsqueeze(1).expand(B * T, N, N)
            invalid = mask_row | mask_col
            logits = logits.masked_fill(invalid, -1e9)

        alpha = F.softmax(logits, dim=-1)
        spatial_features = torch.matmul(alpha, V)
        output_features = self.Theta(spatial_features)

        output_features = output_features.view(B, T, N, D)
        Z_reshaped = Z.view(B, T, N, D)
        out = self.norm(Z_reshaped + output_features)
        out = self.dropout(out)
        out = out.permute(0, 2, 1, 3).contiguous()

        if entity_padding_mask is not None:
            if entity_padding_mask.dim() == 2:
                mask_bn = entity_padding_mask.unsqueeze(-1).unsqueeze(-1)
            else:
                mask_bn = entity_padding_mask.permute(0, 2, 1).unsqueeze(-1)
            if mask_bn.dtype != torch.bool:
                mask_bn = mask_bn.bool()
            out = out.masked_fill(mask_bn, 0.0)

        return out

# ----------------------------------------------------------------------
# 模块 4: 嵌入层 (保持不变)
# ----------------------------------------------------------------------
class TemporalPositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(TemporalPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :, :x.size(2), :]
        return self.dropout(x)

class SpatialPositionalEncoding(nn.Module):
    def __init__(self, num_nodes, d_model, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(num_nodes, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        return self.dropout(x + self.embedding.weight.unsqueeze(0).unsqueeze(2))

# ----------------------------------------------------------------------
# 模块 5: Encoder 层
# ----------------------------------------------------------------------
class EncoderLayer(nn.Module):
    def __init__(self, d_model, temporal_attn, spatial_gnn, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.temporal_attn = temporal_attn
        self.spatial_gnn = spatial_gnn
        self.sublayer_temporal = SublayerConnection(d_model, dropout)
        self.sublayer_spatial = SublayerConnection(d_model, dropout)

    def forward(self, x, temporal_mask, entity_mask, A_social_t=None, edge_features=None):
        B, N, T, D = x.shape
        C = A_social_t.shape[-1]
        E = edge_features.shape[-1]
        A_social_t = A_social_t.reshape(B*T, N, N, C)
        edge_features = edge_features.reshape(B*T, N, N, E)
        
        # 1. Temporal Attention
        x = self.sublayer_temporal(x, lambda x: self.temporal_attn(
            x, x, x, 
            key_padding_mask=temporal_mask
        ))
        
        # 2. Spatial GNN
        x = self.sublayer_spatial(x, lambda x: self.spatial_gnn(
            x, 
            A_prior=A_social_t,
            edge_features=edge_features,
            entity_padding_mask=entity_mask
        ))
        return x

# ----------------------------------------------------------------------
# 模块 6: 完整模型 (Encoder-Only Non-Autoregressive)
# ----------------------------------------------------------------------
class Model(nn.Module):
    
    def __init__(self, args):
        super(Model, self).__init__()
        
        self.num_nodes = args.num_ships
        self.in_features = 7        
        self.out_features = 2       
        self.d_model = args.d_model
        self.num_heads = args.n_heads
        self.num_layers = args.e_layers
        self.dropout = args.dropout
        self.kernel_size = getattr(args, 'kernel_size', 3)
        self.pred_len = args.pred_len
        
        # -- 核心模块 --
        c = copy.deepcopy
        
        # 动力学趋势注意力 (Encoder Mode)
        temporal_attn = KinematicsTrendAttention(
            self.d_model, self.num_heads, dropout=self.dropout)
        
        # 空间 GNN
        spatial_gnn = DynamicSpatialGNN(
            self.d_model, self.num_nodes, dropout=self.dropout)

        # -- 嵌入层 --
        self.src_input_proj = nn.Linear(self.in_features, self.d_model)
        self.next_lane_proj = nn.Sequential(nn.Linear(8, self.d_model), nn.ReLU())
        self.lane_dir_proj = nn.Linear(2, self.d_model)
        
        self.pos_encoder = TemporalPositionalEncoding(self.d_model, self.dropout)
        self.node_encoder = SpatialPositionalEncoding(self.num_nodes, self.d_model, self.dropout)

        # -- Encoder --
        encoder_layer = EncoderLayer(
            self.d_model, c(temporal_attn), c(spatial_gnn), self.dropout)
        self.encoder_layers = clones(encoder_layer, self.num_layers)
        self.encoder_norm = nn.LayerNorm(self.d_model)

        # -- Non-Autoregressive Prediction Head --
        # 将 Encoder 的 Hidden State 映射到 pred_len * out_features
        # 输入: (B, N, d_model) -> 输出: (B, N, pred_len * 2)
        self.output_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model * 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model * 2, self.pred_len * self.out_features)
        )
        
        self.criterion = get_loss_function(args.loss)

    def _compute_pred_deltas(self, x_enc, pred_abs):
        """
        计算预测序列内部的增量
        x_enc:    [B, T_in, N, D]
        pred_abs: [B, T_out, N, 2]
        """
        # 1. 获取历史轨迹的最后一点
        # x_enc 是原始输入 [B, T, N, D]，直接取最后一步 -> [B, 1, N, 2]
        last_known_pos = x_enc[:, -1:, :, :2].to(pred_abs.device) 
        
        # 2. 获取预测序列的前 T-1 个点
        # pred_abs 是 [B, T, N, 2]，取前 T-1 步 -> [B, T-1, N, 2]
        prev_pred_pos = pred_abs[:, :-1, :, :2]
        
        # 3. 拼接得到所有 t 时刻的前一时刻位置
        # 【关键修改】这里不需要 permute，因为 last_known_pos 已经是 [B, 1, N, 2]
        # 它和 prev_pred_pos 在 dim=2 (Node) 上是对齐的
        all_previous_positions = torch.cat([last_known_pos, prev_pred_pos], dim=1)
        
        # 4. 当前时刻绝对位置 - 前一时刻绝对位置 = 增量
        pred_deltas = pred_abs - all_previous_positions
        
        return pred_deltas
        
    def _compute_truth_deltas(self, x_enc, y_truth_abs):
        last_known_pos = x_enc[:, -1:, :, :2] 
        prev_future_pos = y_truth_abs[:, :-1, :, :2]
        all_previous_positions = torch.cat([last_known_pos, prev_future_pos], dim=1)
        y_truth_deltas = y_truth_abs[..., :2] - all_previous_positions
        return y_truth_deltas

    def forward(self, x_enc, y_truth_abs=None, mask_x=None, mask_y=None, A_social_t=None, edge_features=None):
        """
        Encoder-Only Forward
        """
        device = x_enc.device
        
        # 1. 准备输入
        mask_x_permuted = mask_x.permute(0, 2, 1) # [B, N, T] -> [B, T, N] ? 需确认 mask 维度
        # 假设 mask_x 是 [B, N, T] (True=Valid), TrendAwareAttention 需要 [B, N, T] (True=Padding)
        # 通常 mask_x 在外部已经处理好, 这里假设 mask_x: [B, N, T]
        # Attention mask: True 为屏蔽 (Padding)
        temporal_padding_mask_enc = ~mask_x # [B, N, T]
        entity_mask = mask_x.any(dim=-1) # [B, N], 只要该节点有一个时间步有效，就算有效
        entity_padding_mask = ~entity_mask # [B, N] True=Padding

        # 1. 处理输入数据 (已修正)
        # [B, T, N, D] -> [B, N, T, D]
        seq_x = x_enc[..., :7].to(device)
        seq_x = seq_x.permute(0, 2, 1, 3) 
        
        # -----------------------------------------------------------
        # 【新增修正】处理 Mask，必须跟随数据一起变维度
        # -----------------------------------------------------------
        if mask_x is not None:
            mask_x = mask_x.to(device)
            # 假设 mask_x 原始是 [B, T, N] (Batch, Time, Node)
            # 我们需要它变成 [B, N, T] 以匹配 TrendAwareAttention 的需求
            mask_x_permuted = mask_x.permute(0, 2, 1) 
            
            # 确保是 Bool 类型
            if mask_x_permuted.dtype != torch.bool:
                mask_x_permuted = mask_x_permuted.bool()
            
            # 生成 Attention 用的 Mask (True 表示 Padding/无效)
            temporal_padding_mask_enc = ~mask_x_permuted
            
            # 生成 GNN 用的 Entity Mask (只要该节点在任意时刻有效，就算有效)
            # 原始 mask_x 是 [B, T, N]，沿着 T(dim=1) 维度求 any
            # 结果为 [B, N]，表示哪些船是存在的
            entity_mask = mask_x.any(dim=1) 
            entity_padding_mask = ~entity_mask 
        else:
            temporal_padding_mask_enc = None
            entity_padding_mask = None
        # -----------------------------------------------------------

        enc_in = self.src_input_proj(seq_x) 
        
        # ... (中间的 next_lane / lane_dir 处理保持上一轮的 permute 修改) ...
        # next_lane_onehot 和 lane_dir_feats 也要记得 permute !
        next_lane_onehot = x_enc[..., 7:15].to(device)
        lane_dir_feats = x_enc[..., 15:17].to(device)

        if next_lane_onehot is not None:
            # [B, T, N, 8] -> [B, N, T, 8]
            next_lane_perm = next_lane_onehot.permute(0, 2, 1, 3).float()
            enc_in = enc_in + self.next_lane_proj(next_lane_perm)
            
        if lane_dir_feats is not None:
            # [B, T, N, 2] -> [B, N, T, 2]
            lane_dir_perm = lane_dir_feats.permute(0, 2, 1, 3).float()
            enc_in = enc_in + self.lane_dir_proj(lane_dir_perm)

        enc_in = self.node_encoder(enc_in) 
        enc_in = self.pos_encoder(enc_in)  
        
        # 2. Encoder
        memory = enc_in
        for layer in self.encoder_layers:
            memory = layer(
                memory, 
                # 这里传入修正后的 Mask
                temporal_mask=temporal_padding_mask_enc, 
                entity_mask=entity_padding_mask,
                A_social_t=A_social_t,
                edge_features=edge_features
            )
        memory = self.encoder_norm(memory) # [B, N, T, D]

        # 3. Non-Autoregressive Prediction Head
        # 取最后一个时间步的特征作为预测基础
        # memory: [B, N, T, D] -> [B, N, D]
        last_step_feature = memory[:, :, -1, :]
        
        # 通过 MLP 直接预测未来所有步的偏移量
        # output_deltas_flat: [B, N, Pred_Len * 2]
        output_deltas_flat = self.output_head(last_step_feature)
        
        # Reshape: [B, N, Pred_Len, 2]
        output_deltas = output_deltas_flat.view(
            output_deltas_flat.size(0), 
            output_deltas_flat.size(1), 
            self.pred_len, 
            2
        )
        
        # 4. 恢复绝对坐标
        # 获取历史最后一个观测点的绝对坐标
        # x_enc: [B, N, T, F] -> [B, N, -1, :2]
        last_pos = seq_x[:, :, -1:, :2] # [B, N, 1, 2]
        
        # 累加增量得到绝对坐标序列
        # P_t = P_last + sum(delta_1...delta_t)
        # 注意：这里 MLP 输出的可以是"相对于上一时刻的增量"，也可以是"相对于起点的累计增量"
        # 为了让模型容易学习，我们让 MLP 输出"相对于起点的累计偏移 (Cumulative Offsets)"
        # 这样 P_future = P_last + Output
        # 如果输出的是瞬时速度 (Step Deltas)，则需要 torch.cumsum
        
        # 这里采用: MLP 输出的是 Step Deltas (每一步的位移)，通过 cumsum 得到轨迹
        # 这样能保证轨迹的连续性
        cumulative_deltas = torch.cumsum(output_deltas, dim=2) # 沿时间轴 cumsum
        output_absolute = last_pos + cumulative_deltas # [B, N, Pred_Len, 2]

        # Reshape to [B, Pred_Len, N, 2] to match original output format
        output_absolute = output_absolute.permute(0, 2, 1, 3) 
        
        if mask_y is not None:
            final_mask = mask_y.unsqueeze(-1).float()
            return output_absolute * final_mask
        
        return output_absolute

    def loss(self, pred, y_truth_abs, x_enc, mask_y, iter=None, epoch=None):
        """
        Loss 计算
        pred: [B, T, N, 2] (绝对坐标)
        y_truth_abs: [B, T, N, 2] (绝对坐标)
        """
        y_truth_abs = y_truth_abs[..., :2]
        
        if mask_y.dtype == torch.float:
            mask_y_bool = mask_y.bool()
        else:
            mask_y_bool = mask_y

        # 1. 绝对坐标 Loss (MAE)
        loss_absolute = get_loss_function('mae')(
            pred[mask_y_bool],
            y_truth_abs[mask_y_bool]
        )

        # 2. 增量 Loss (MSE) - 约束形状和速度
        y_truth_deltas = self._compute_truth_deltas(x_enc, y_truth_abs)
        pred_deltas = self._compute_pred_deltas(x_enc, pred)
        
        loss_delta = self.criterion(
            pred_deltas[mask_y_bool], 
            y_truth_deltas[mask_y_bool]
        )
        
        loss_mse = get_loss_function('mse')(
            pred[mask_y_bool],
            y_truth_abs[mask_y_bool]
        )

        # 组合 Loss
        back_loss = 0.1 * loss_delta + loss_absolute 
        
        if (iter is not None and epoch is not None) and (iter + 1) % 100 == 0:
            print(f"\titers: {iter + 1}, epoch: {epoch + 1} |  Loss_Delta: {loss_delta.item():.7f}, Loss_Abs: {loss_absolute.item():.7f}, Loss_Abs_MSE: {loss_mse.item():.7f}")

        return back_loss, loss_mse