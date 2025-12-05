import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
import numpy as np
from utils.get_loss_function import get_loss_function

"""
V3.3.0 (Encoder-Only + Direct Absolute Prediction)
- 架构: Encoder-Only (Non-Autoregressive)
- 输出: 直接预测绝对坐标 (Absolute Coordinates), 不再预测增量
- 修复: 输入维度自动转置，解决 [B,T,N] vs [B,N,T] 冲突
"""

# ----------------------------------------------------------------------
# 模块 1: 帮助函数
# ----------------------------------------------------------------------
def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

# ----------------------------------------------------------------------
# 模块 2: 时间趋势感知注意力 (Encoder Mode)
# ----------------------------------------------------------------------
class TrendAwareAttention(nn.Module):
    def __init__(self, d_model, num_heads, kernel_size=3, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.kernel_size = kernel_size

        self.conv_q = nn.Conv1d(d_model, d_model, kernel_size, padding='same')
        self.conv_k = nn.Conv1d(d_model, d_model, kernel_size, padding='same')

        self.linear_v = nn.Linear(d_model, d_model)
        self.linear_out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, key_padding_mask=None):
        # Input: [B, N, T, D]
        B, N, T_q, D = query.shape
        _, _, T_k, _ = key.shape
        
        q_conv_in = query.reshape(B*N, T_q, D).transpose(1, 2)
        k_conv_in = key.reshape(B*N, T_k, D).transpose(1, 2)
        
        q_conv_out = self.conv_q(q_conv_in).transpose(1, 2) 
        k_conv_out = self.conv_k(k_conv_in).transpose(1, 2)
        
        Q = q_conv_out.reshape(B, N, T_q, self.num_heads, self.d_k).transpose(2, 3)
        K = k_conv_out.reshape(B, N, T_k, self.num_heads, self.d_k).transpose(2, 3)
        V = self.linear_v(value).view(B, N, T_k, self.num_heads, self.d_k).transpose(2, 3)

        # Q, K, V: [B, N, H, T, D_k]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # mask 必须是 [B, N, 1, 1, T] 才能广播到 [B, N, H, T, T]
        if key_padding_mask is not None:
            mask = key_padding_mask.unsqueeze(2).unsqueeze(3)
            scores = scores.masked_fill(mask, -1e9)
            
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        x = torch.matmul(attn, V)
        x = x.transpose(2, 3).contiguous().view(B, N, T_q, self.d_model)

        return self.linear_out(x)

# ----------------------------------------------------------------------
# 模块 3: 动态空间 GNN
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
        # Input: [B, N, T, D]
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
            # entity_mask: [B, N] -> [B*T, N]
            mask_bt = entity_padding_mask.unsqueeze(1).expand(B, T, N).contiguous().view(B*T, N)
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
        out = out.permute(0, 2, 1, 3).contiguous() # -> [B, N, T, D]

        if entity_padding_mask is not None:
            # [B, N] -> [B, N, 1, 1]
            mask_bn = entity_padding_mask.unsqueeze(-1).unsqueeze(-1)
            if mask_bn.dtype != torch.bool:
                mask_bn = mask_bn.bool()
            out = out.masked_fill(mask_bn, 0.0)

        return out

# ----------------------------------------------------------------------
# 模块 4: 嵌入层
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
        pe = pe.unsqueeze(0).unsqueeze(0) # [1, 1, Max_T, D]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [B, N, T, D]
        # pe: [1, 1, T, D] -> Broadcasts correctly
        x = x + self.pe[:, :, :x.size(2), :]
        return self.dropout(x)

class SpatialPositionalEncoding(nn.Module):
    def __init__(self, num_nodes, d_model, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(num_nodes, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        # x: [B, N, T, D]
        # weight: [N, D] -> [1, N, 1, D]
        # Expects x dimension 1 to be Nodes
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
        # Flatten time for GNN input logic inside the module
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
# 模块 6: 完整模型 (Encoder-Only + Direct Abs Prediction)
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
        
        temporal_attn = TrendAwareAttention(
            self.d_model, self.num_heads, self.kernel_size, dropout=self.dropout)
        
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

        # -- Regression Head --
        # 直接输出 [pred_len * 2] 的数值
        self.output_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model * 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model * 2, self.pred_len * self.out_features)
        )
        
        self.criterion = get_loss_function(args.loss)

    def _compute_truth_deltas(self, x_enc, y_truth_abs):
        # x_enc: [B, T, N, D]
        # y_truth_abs: [B, T, N, 2]
        last_known_pos = x_enc[:, -1:, :, :2] 
        prev_future_pos = y_truth_abs[:, :-1, :, :2]
        all_previous_positions = torch.cat([last_known_pos, prev_future_pos], dim=1)
        y_truth_deltas = y_truth_abs[..., :2] - all_previous_positions
        return y_truth_deltas

    def _compute_pred_deltas(self, x_enc, pred_abs):
        """
        计算预测出的绝对坐标中隐含的增量（用于辅助Loss）
        x_enc:    [B, T_in, N, D]
        pred_abs: [B, T_out, N, 2]
        """
        # 1. 历史最后一点 [B, 1, N, 2]
        last_known_pos = x_enc[:, -1:, :, :2].to(pred_abs.device) 
        
        # 2. 预测序列的前 T-1 点 [B, T-1, N, 2]
        prev_pred_pos = pred_abs[:, :-1, :, :2]
        
        # 3. 拼接
        all_previous_positions = torch.cat([last_known_pos, prev_pred_pos], dim=1)
        
        # 4. 差分
        pred_deltas = pred_abs - all_previous_positions
        return pred_deltas

    def forward(self, x_enc, y_truth_abs=None, mask_x=None, mask_y=None, A_social_t=None, edge_features=None):
        device = x_enc.device
        
        # ------------------------------------------------------------------
        # 1. 关键：维度处理 [B, T, N, D] -> [B, N, T, D]
        # ------------------------------------------------------------------
        # 假设 x_enc 原始输入是 [B, T, N, D] (例如 Batch, 8, 17, Feature)
        seq_x = x_enc[..., :7].to(device)
        seq_x = seq_x.permute(0, 2, 1, 3) # -> [B, N, T, D]
        
        # Mask 处理: mask_x [B, T, N] -> [B, N, T]
        if mask_x is not None:
            mask_x = mask_x.to(device)
            mask_x_permuted = mask_x.permute(0, 2, 1) # [B, N, T]
            
            if mask_x_permuted.dtype != torch.bool:
                mask_x_permuted = mask_x_permuted.bool()
            
            # True = Padding
            temporal_padding_mask_enc = ~mask_x_permuted
            
            # GNN Mask [B, N]
            entity_mask = mask_x.any(dim=1) 
            entity_padding_mask = ~entity_mask 
        else:
            temporal_padding_mask_enc = None
            entity_padding_mask = None

        # ------------------------------------------------------------------
        # 2. 模型主体
        # ------------------------------------------------------------------
        enc_in = self.src_input_proj(seq_x) 
        
        # 额外特征处理 (同样需要 Permute)
        next_lane_onehot = x_enc[..., 7:15].to(device) # [B, T, N, 8]
        lane_dir_feats = x_enc[..., 15:17].to(device)  # [B, T, N, 2]

        if next_lane_onehot is not None:
            # [B, T, N, 8] -> [B, N, T, 8]
            next_lane_perm = next_lane_onehot.permute(0, 2, 1, 3).float()
            enc_in = enc_in + self.next_lane_proj(next_lane_perm)
            
        if lane_dir_feats is not None:
            # [B, T, N, 2] -> [B, N, T, 2]
            lane_dir_perm = lane_dir_feats.permute(0, 2, 1, 3).float()
            enc_in = enc_in + self.lane_dir_proj(lane_dir_perm)

        # 此时 enc_in: [B, N, T, D]
        # SpatialPositionalEncoding 正确加在 dim=1 (N)
        enc_in = self.node_encoder(enc_in) 
        # TemporalPositionalEncoding 正确加在 dim=2 (T)
        enc_in = self.pos_encoder(enc_in)  
        
        # Encoder Forward
        memory = enc_in
        for layer in self.encoder_layers:
            memory = layer(
                memory, 
                temporal_mask=temporal_padding_mask_enc, 
                entity_mask=entity_padding_mask,
                A_social_t=A_social_t,
                edge_features=edge_features
            )
        memory = self.encoder_norm(memory) # [B, N, T, D]

        # ------------------------------------------------------------------
        # 3. 直接预测绝对坐标 (Direct Absolute Prediction)
        # ------------------------------------------------------------------
        # 取最后一个时间步特征
        last_step_feature = memory[:, :, -1, :] # [B, N, D]
        
        # MLP 直接输出目标值
        # output_flat: [B, N, Pred_Len * 2]
        output_flat = self.output_head(last_step_feature)
        
        # Reshape: [B, N, Pred_Len, 2]
        output_absolute = output_flat.view(
            output_flat.size(0), 
            output_flat.size(1), 
            self.pred_len, 
            2
        )
        
        # 4. 调整回 [B, T, N, 2] 以匹配 Ground Truth
        output_absolute = output_absolute.permute(0, 2, 1, 3) 
        
        if mask_y is not None:
            final_mask = mask_y.unsqueeze(-1).float()
            return output_absolute * final_mask
        
        return output_absolute

    def loss(self, pred, y_truth_abs, x_enc, mask_y, iter=None, epoch=None):
        """
        Loss 计算
        pred: [B, T, N, 2] (模型直接预测的绝对坐标)
        """
        y_truth_abs = y_truth_abs[..., :2]
        
        if mask_y.dtype == torch.float:
            mask_y_bool = mask_y.bool()
        else:
            mask_y_bool = mask_y

        # 1. 主 Loss: 绝对坐标准确度 (MAE)
        loss_absolute = get_loss_function('mae')(
            pred[mask_y_bool],
            y_truth_abs[mask_y_bool]
        )

        # 2. 辅助 Loss: 隐含增量一致性 (MSE)
        # 即使直接预测绝对坐标，强制其隐含的速度/增量符合物理规律也能帮助收敛
        y_truth_deltas = self._compute_truth_deltas(x_enc, y_truth_abs)
        pred_deltas = self._compute_pred_deltas(x_enc, pred)
        
        loss_delta = self.criterion(
            pred_deltas[mask_y_bool], 
            y_truth_deltas[mask_y_bool]
        )
        
        # 显示用的 MSE
        loss_mse = get_loss_function('mse')(
            pred[mask_y_bool],
            y_truth_abs[mask_y_bool]
        )

        # 组合 Loss: 主 Loss + 0.1 * 物理辅助 Loss
        back_loss = loss_absolute + 0.1 * loss_delta
        
        if (iter is not None and epoch is not None) and (iter + 1) % 100 == 0:
            print(f"\titers: {iter + 1}, epoch: {epoch + 1} |  Loss_Delta: {loss_delta.item():.7f}, Loss_Abs: {loss_absolute.item():.7f}, Loss_Abs_MSE: {loss_mse.item():.7f}")

        return back_loss, loss_mse