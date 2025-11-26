from shapely.geometry import LineString, Point
import pandas as pd
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import warnings

warnings.filterwarnings('ignore')

# ===== 航道文件读取 =====
def load_lane_table(path: str):
    # 判断是否为execl或者csv文件：
    ext = os.path.splitext(path)[1].lower()
    print(path)
    if ext in [".xlsx", ".xls"]:
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path, encoding="utf-8")


    df = df.rename(columns={
        df.columns[0]: "lane_name",   # 名称
        df.columns[1]: "lane_dir",    # NS / SN / EW / WE
        df.columns[2]: "lane_role",   # center / upper / lower
        df.columns[3]: "seq",         # 序号
        df.columns[4]: "lon",         # 经度
        df.columns[5]: "lat",         # 纬度
    })
    print("[Lane] load_lane_table OK, type(df) =", type(df))
    print(df.head())
    return df

# ===== 构建航道几何 =====
def build_lanes(df: pd.DataFrame):
    lanes = {}
    for (lane_name, lane_dir, lane_role), g in df.groupby(["lane_name", "lane_dir", "lane_role"]):
        if str(lane_dir).upper() in ("EW", "WE"):
            g = g.sort_values("lon")
        else:
            g = g.sort_values("lat")
        coords = list(zip(g["lon"].values, g["lat"].values))
        if len(coords) < 2:
            continue
        line = LineString(coords)
        entry = lanes.setdefault(lane_name, {"dir": lane_dir})
        entry[lane_role] = line
    print("[Lane] build_lanes OK, lanes keys:", list(lanes.keys()))
    return lanes

# ===== 找最近航道 + 计算 s / d =====
def find_nearest_lane(lanes, lon, lat):
    p = Point(lon, lat)
    best_name, best_entry, best_dist = None, None, 1e9
    for name, info in lanes.items():
        center = info.get("center", None)
        if center is None:
            continue
        d = p.distance(center)
        if d < best_dist:
            best_dist = d
            best_name = name
            best_entry = info
    return best_name, best_entry, best_dist

def lane_features_for_point(p: Point, lane_entry):
    line = lane_entry["center"]
    s = line.project(p)
    s_norm = s / max(line.length, 1e-6)

    proj = line.interpolate(s)
    d = p.distance(proj)

    x0, y0 = proj.x, proj.y
    s2 = min(s + 1.0, line.length)
    ahead = line.interpolate(s2)
    dx, dy = ahead.x - x0, ahead.y - y0
    vx, vy = p.x - x0, p.y - y0
    sign = np.sign(-dy * vx + dx * vy)
    d_signed = d * sign

    return s_norm, d_signed




class ShipTrajectoryDataset(Dataset):
    """
    船舶轨迹预测数据集
    
    支持从npz文件加载预处理好的数据
    数据格式：
        - X: [B, T_in, N, D] 船舶历史轨迹
        - y: [B, T_out, N, D] 需要被预测的轨迹
        - X_mask: [B, T_in, N] 输入掩码
        - y_mask: [B, T_out, N] 输出掩码
        - ship_counts: [B] 每个样本的船舶数量
        - global_ids: [B, N] 船舶MMSI编号
    """
    
    def __init__(
        self,
        root_path,
        data_path,
        flag='train',
        size=None,
        num_ships=17,
        num_features=4,
        scale=True,
        scale_type='standard',
        predict_position_only=True,
        lane_table_path=None
    ):
        """
        Args:
            root_path: 数据根目录
            data_path: npz文件路径
            flag: 'train', 'val', 'test'
            size: [seq_len, pred_len] 输入和输出序列长度
            num_ships: 船舶数量（固定为17）
            num_features: 特征数量（经度、纬度、COG、SOG = 4）
            scale: 是否进行数据归一化
            scale_type: 归一化类型 'standard' 或 'minmax'
            predict_position_only: 是否只预测经纬度（True时只计算经纬度的损失）
        """
        # 基本参数
        self.root_path = root_path
        self.data_path = data_path
        self.flag = flag
        self.num_ships = num_ships
        self.num_features = num_features
        self.scale = scale
        self.scale_type = scale_type
        self.predict_position_only = predict_position_only
        self.social_sigma = 1.0
        # [新] 避碰规则参数 (TCPA/DCPA 阈值)
        self.tcpa_threshold = 300.0 # 例如 300秒 (5分钟)
        self.dcpa_threshold = 1000.0 # 例如 1000米

        # 航道几何特征
        self.lanes = None
        if lane_table_path is not None:
            df_lanes = load_lane_table(lane_table_path)
            self.lanes = build_lanes(df_lanes)
            print(f"[lane] Loaded lanes: {list(self.lanes.keys())}")

        
        # 序列长度
        if size is None:
            self.seq_len = 8
            self.pred_len = 12
        else:
            self.seq_len = size[0]
            self.pred_len = size[1]
        
        # 加载数据
        self.__read_data__()
    
    def __read_data__(self):
        """从npz文件读取数据"""
        # 构建完整路径
        full_path = os.path.join(self.root_path, self.data_path)
        
        print(f'Loading {self.flag} data from: {full_path}')
        
        # 加载npz文件
        data = np.load(full_path)
        
        # 读取所有字段
        self.data_x = data['X']  # [B, T_in, N, D]
        self.data_y = data['y']  # [B, T_out, N, 2]
        self.mask_x = data['X_mask']  # [B, T_in, N]
        self.mask_y = data['y_mask']  # [B, T_out, N]
        self.ship_counts = data['ship_counts']  # [B]
        self.global_ids = data['global_ids']  # [B, N]
        
        # 打印数据信息
        print(f'{self.flag} dataset loaded:')
        print(f'  X shape: {self.data_x.shape}')
        print(f'  y shape: {self.data_y.shape}')
        print(f'  X_mask shape: {self.mask_x.shape}')
        print(f'  y_mask shape: {self.mask_y.shape}')
        print(f'  ship_counts shape: {self.ship_counts.shape}')
        print(f'  global_ids shape: {self.global_ids.shape}')
        print(f'  Number of samples: {len(self.data_x)}')
        print(f'  Average ship count: {self.ship_counts.mean():.2f}')
        
        # 验证数据维度
        assert self.data_x.shape[1] == self.seq_len, f"X time dimension mismatch: {self.data_x.shape[1]} != {self.seq_len}"
        assert self.data_y.shape[1] == self.pred_len, f"y time dimension mismatch: {self.data_y.shape[1]} != {self.pred_len}"
        assert self.data_x.shape[2] == self.num_ships, f"X ship dimension mismatch: {self.data_x.shape[2]} != {self.num_ships}"
        assert self.data_y.shape[2] == self.num_ships, f"y ship dimension mismatch: {self.data_y.shape[2]} != {self.num_ships}"
        assert self.data_x.shape[3] == self.num_features, f"X feature dimension mismatch: {self.data_x.shape[3]} != {self.num_features}"
        
        # 数据归一化
        if self.scale:
            self.__normalize_data__()
    
    def _transform_features(self, x, mask):
        """
        将原始的 4维特征 [Lat, Lon, SOG, COG] 转换为 5维特征 [Lat, Lon, SOG, Cos, Sin]。
        
        Args:
            raw_data: 归一化后的数据 [L, 4] 或 [L, N, 4]
            
        Returns:
            new_data: 转换后的数据 [L, 5] 或 [L, N, 5]
        """
        assert mask is not None, "Mask must be provided"
        # 分割数据
        feats_others = x[..., :3]
        cog_scaled = x[..., 3]

        # 1. 反归一化COG
        if self.scale_type == 'standard':
            cog_phys = cog_scaled * self.std[3] + self.mean[3]
        elif self.scale_type == 'minmax':
            cog_phys = cog_scaled * (self.max_val[3] - self.min_val[3]) + self.min_val[3]
        # 2. 计算 Cos 和 Sin 分量
        cog_rad = np.deg2rad(cog_phys)
        cog_cos = np.cos(cog_rad)
        cog_sin = np.sin(cog_rad)
        # 3. 拼接新特征

        # print('feats_others shape:', feats_others.shape)
        # print('cog_cos shape:', cog_cos.shape)
        # print('cog_sin shape:', cog_sin.shape)

        if mask is not None:
            # 确保 mask 是 bool 类型
            if mask.dtype != bool: 
                mask = mask.astype(bool)

        new_feats = np.concatenate([feats_others, cog_cos[..., np.newaxis], cog_sin[..., np.newaxis]], axis=-1)
        
        new_feats = new_feats * mask[..., np.newaxis]  # 只保留有效位置的数据

        return new_feats.astype(np.float32)



    def __normalize_data__(self):
        """
        数据归一化
        只对训练集计算统计量，验证集和测试集使用训练集的统计量
        """
        if self.flag == 'train':
            # 对每个特征分别计算统计量
            # 只使用有效数据点（mask==1）计算统计量
            self.mean = np.zeros(self.num_features)
            self.std = np.zeros(self.num_features)
            self.min_val = np.zeros(self.num_features)
            self.max_val = np.zeros(self.num_features)
            
            for feat_idx in range(self.num_features):
                # 获取所有有效数据点
                valid_data = self.data_x[:, :, :, feat_idx][self.mask_x == 1]
                
                if self.scale_type == 'standard':
                    self.mean[feat_idx] = valid_data.mean()
                    self.std[feat_idx] = valid_data.std()
                elif self.scale_type == 'minmax':
                    self.min_val[feat_idx] = valid_data.min()
                    self.max_val[feat_idx] = valid_data.max()

            #############
            if self.scale_type == 'standard':
                self.std = np.where(self.std < 1e-8, 1.0, self.std)
            elif self.scale_type == 'minmax':
                span = self.max_val - self.min_val
                span = np.where(span < 1e-8, 1.0, span)
                self.max_val = self.min_val + span
            #############

            # 保存归一化参数
            self.scaler_params = {
                'mean': self.mean,
                'std': self.std,
                'min_val': self.min_val,
                'max_val': self.max_val,
                'scale_type': self.scale_type
            }
            
            np.save(os.path.join(self.root_path, 'scaler_params.npy'), self.scaler_params)

            print(f'\nNormalization parameters ({self.scale_type}):')
            feature_names = ['Longitude', 'Latitude', 'SOG', 'COG']
            for i, name in enumerate(feature_names):
                if self.scale_type == 'standard':
                    print(f'  {name}: mean={self.mean[i]:.4f}, std={self.std[i]:.4f}')
                else:
                    print(f'  {name}: min={self.min_val[i]:.4f}, max={self.max_val[i]:.4f}')
            
            # 归一化数据
            self._apply_normalization(self.data_x, mask=self.mask_x)
            self._apply_normalization(self.data_y, mask=self.mask_y)

        else:
            # 验证集和测试集需要加载训练集的归一化参数
            # 这里假设训练时会保存归一化参数
            scaler_path = os.path.join(self.root_path, 'scaler_params.npy')
            if os.path.exists(scaler_path):
                self.scaler_params = np.load(scaler_path, allow_pickle=True).item()
                self.mean = self.scaler_params['mean']
                self.std = self.scaler_params['std']
                self.min_val = self.scaler_params['min_val']
                self.max_val = self.scaler_params['max_val']
                self.scale_type = self.scaler_params['scale_type']
                
                print(f'\nLoaded normalization parameters from: {scaler_path}')
                
                # 归一化数据
                self._apply_normalization(self.data_x, mask=self.mask_x)
                self._apply_normalization(self.data_y, mask=self.mask_y)
            else:
                print(f'\nWarning: Scaler parameters not found at {scaler_path}')
                print('Using data without normalization')
    
    def _apply_normalization_old(self, data):
        """
        应用归一化
        
        Args:
            data: [B, T, N, D] 需要归一化的数据
        """
        for feat_idx in range(self.num_features):
            if self.scale_type == 'standard':
                # 标准化: (x - mean) / std
                data[:, :, :, feat_idx] = (data[:, :, :, feat_idx] - self.mean[feat_idx]) / (self.std[feat_idx] + 1e-8)
            elif self.scale_type == 'minmax':
                # 最小-最大归一化: (x - min) / (max - min)
                data[:, :, :, feat_idx] = (data[:, :, :, feat_idx] - self.min_val[feat_idx]) / (self.max_val[feat_idx] - self.min_val[feat_idx] + 1e-8)

    def _apply_normalization(self, data, mask):
        """
            只对有效位置做归一化；padding 位置保持原值（通常是 0），
            后续在模型前处理阶段再用 pad_token 替换。
            data: [B, T, N, D']
            mask: [B, T, N]  (1=有效, 0=padding)
        """
        B, T, N, Dp = data.shape
        F = min(self.num_features, Dp)  # y 可能只有2维，防越界
        m = mask[...].astype(np.bool_)        # [B,T,N,1]
        print('mask shape:', m.shape)
        if self.scale_type == 'standard':
            for f in range(F):
                data[..., f] = np.where(
                    m,
                    (data[..., f] - self.mean[f]) / (self.std[f] + 1e-8),
                    data[..., f]
                )
        elif self.scale_type == 'minmax':
            for f in range(F):
                data[..., f] = np.where(
                    m,
                    (data[..., f] - self.min_val[f]) / (self.max_val[f] - self.min_val[f] + 1e-8),
                    data[..., f]
                )
    
    def inverse_transform(self, data):
        """
        反归一化（用于预测后还原真实值）
        
        Args:
            data: [B, T, N, D] or [B, T, N, 2] 归一化后的数据
        
        Returns:
            反归一化后的数据
        """
        if not self.scale:
            return data
        
        data_copy = data.copy()
        num_features_to_inverse = data.shape[-1]
        
        for feat_idx in range(num_features_to_inverse):
            if self.scale_type == 'standard':
                data_copy[..., feat_idx] = data_copy[..., feat_idx] * self.std[feat_idx] + self.mean[feat_idx]
            elif self.scale_type == 'minmax':
                data_copy[..., feat_idx] = data_copy[..., feat_idx] * (self.max_val[feat_idx] - self.min_val[feat_idx]) + self.min_val[feat_idx]
        
        return data_copy
    
    def save_scaler_params(self, save_path=None):
        """
        保存归一化参数（仅用于训练集）
        
        Args:
            save_path: 保存路径，默认为 root_path/scaler_params.npy
        """
        if self.flag == 'train' and self.scale:
            if save_path is None:
                save_path = os.path.join(self.root_path, 'scaler_params.npy')
            
            np.save(save_path, self.scaler_params)
            print(f'Scaler parameters saved to: {save_path}')
    
    def __getitem__(self, index):
        """
        获取单个样本
        
        Returns:
            seq_x: [T_in, N, D] 输入序列
            seq_y: [T_out, N, D] 目标序列
            seq_x_mask: [T_in, N] 输入mask
            seq_y_mask: [T_out, N] 输出mask
            ship_count: 标量，船舶数量
            global_id: [N] MMSI编号
        """
        seq_x = self.data_x[index]  # [T_in, N, D]
        seq_y = self.data_y[index]  # [T_out, N, D]
        seq_x_mask = self.mask_x[index]  # [T_in, N]
        seq_y_mask = self.mask_y[index]  # [T_out, N]
        ship_count = self.ship_counts[index]  # scalar
        global_id = self.global_ids[index]  # [N]

        # 计算 social matrix 
        # A_social = self._build_social_graph(seq_x, mask=seq_x_mask)  # [T_in, N, N]
        # A_social_dec = self._build_social_graph(seq_y)

        # 计算 "语义" 社会力矩阵 社会影响力 + COLREGs 规则
        A_social = self._build_semantic_social_fusion_matrix(seq_x)
        edge_features = self._build_edge_features(seq_x, mask=seq_x_mask)

        # ===== 新增：航道特征 [T_in,N,2] =====
        if self.lanes is not None:
            lane_feats = self._build_lane_features(seq_x, seq_x_mask)  # [T_in,N,2]
        else:
            lane_feats = np.zeros((seq_x.shape[0], seq_x.shape[1], 2), dtype=np.float32)

        seq_x = self._transform_features(seq_x, seq_x_mask)
        seq_y = self._transform_features(seq_y, seq_y_mask)
        
        seq_x_final = np.concatenate([seq_x, lane_feats], axis=-1).astype(np.float32)



        return seq_x_final, seq_y, seq_x_mask, seq_y_mask, ship_count, global_id, A_social, edge_features
        # return seq_x, seq_y, seq_x_mask, seq_y_mask, ship_count, global_id, A_social
    
    def __len__(self):
        return len(self.data_x)

    def _build_social_graph(self, x_data, mask=None):
        """
        构建社交图（基于船舶之间的距离）
        这里可以实现基于距离的邻接矩阵构建
        """
        """
        根据 "社会力" 公式 (2) 构建动态邻接矩阵。
        
        Args:
            x_data (np.ndarray): 轨迹数据。Shape: [T_seq, N, D_in]
        
        Returns:
            np.ndarray: A_social。Shape: [T_seq, N, N]
        """
        T_seq, N, D_in = x_data.shape
        pos_data = x_data[:, :, :2]
        speed_data = x_data[:, :, 2]
        heading_data = x_data[:, :, 3]

        # 1. 计算速度向量(v_x, v_y)
        heading_rad = np.deg2rad(heading_data)
        v_x = speed_data * np.sin(heading_rad)
        v_y = speed_data * np.cos(heading_rad)
        vel_vectors = np.stack((v_x, v_y), axis=-1)  # [T_seq, N, 2]

        # 准备广播
        # [T, N, 1, 2]
        pos_i = np.expand_dims(pos_data, axis=2)
        vel_i = np.expand_dims(vel_vectors, axis=2)
        # [T, 1, N, 2]
        pos_j = np.expand_dims(pos_data, axis=1)
        vel_j = np.expand_dims(vel_vectors, axis=1)

        # 2. 计算 ||pos_i - pos_j|| （位置距离）
        # [T, N, N, 2] -> [T, N, N]
        # (注意: 这是 (lon, lat) 上的欧几里得距离, 
        #  对于小范围是合理的近似)
        dist_matrix = np.linalg.norm(pos_i - pos_j, axis=-1) + 1e-8  # 防止除零

        # 3. 计算 ||v_i - v_j||^2 (速度差异)
        # [T, N, N, 2] -> [T, N, N]
        vel_diff_sq = np.sum((vel_i - vel_j), axis=-1)

        # 4. 计算 A_social
        # a, 计算指数项
        # exp(- ||v_i - v_j||^2 / (2*sigma^2))
        vel_term = np.exp(- vel_diff_sq / (2 * (self.social_sigma ** 2)))  # sigma=1.0 可调

        # b, 计算 距离项
        dist_term = 1.0 / (dist_matrix + 1e-9)

        # c, .构建 A_social
        A_social = vel_term * dist_term

        # 处理幽灵船的无效交互
        if mask is not None:
            if mask.dtype != bool:
                mask = mask.astype(bool)
            
            mask_i = np.expand_dims(mask, axis=2)  # [T, N, 1]  代表船 i
            mask_j = np.expand_dims(mask, axis=1)  # [T, 1, N]  代表船 j
            combined_mask = (mask_i & mask_j)  # [T, N, N]  当且仅当 t 时刻的船 i 和船 j *都* 是有效的
            A_social = A_social * combined_mask

        # d. 处理对角线(i=j)
        diag_idx = np.arange(N)
        A_social[:, diag_idx, diag_idx] = 0.0

        return A_social # [T_seq, N, N]

    def _build_semantic_social_matrix(self, x_data, mask=None):
        """
        构建 "语义" 社会力矩阵 (基于 COLREGs 规则)
        返回形状: [T, N, N, 4]
        通道: 0=Head-on, 1=Starboard_Cross, 2=Port_Cross, 3=Overtaking
        """
        T, N, D = x_data.shape
        
        # 1. 反归一化 (必须在物理空间计算几何关系)
        #    假设前4维是 [lat, lon, speed, course]
        x_phys = x_data * self.scaler_params['mean'][:D] + self.scaler_params['std'][:D]
        
        # 提取物理量
        # 注意: 经纬度直接算欧氏距离在小范围是近似，严谨应投影。
        # 这里为了效率，假设 lat/lon 也就是平面坐标 (或者数据已经是投影后的 x,y)
        pos_phys = x_phys[..., :2]      # [T, N, 2] (x, y)
        speed_phys = x_phys[..., 2]     # [T, N]
        course_phys = x_phys[..., 3]    # [T, N] (度)

        # 2. 计算速度向量 (vx, vy)
        #    假设 Course 是以北为0，顺时针 (0~360)
        #    转为数学弧度: math_angle = 90 - course
        course_rad = np.deg2rad(90 - course_phys) 
        vx = speed_phys * np.cos(course_rad)
        vy = speed_phys * np.sin(course_rad)
        vel_phys = np.stack([vx, vy], axis=-1) # [T, N, 2]

        # 3. 准备广播维度
        # i: 本船, j: 他船
        pos_i = np.expand_dims(pos_phys, axis=2)   # [T, N, 1, 2]
        pos_j = np.expand_dims(pos_phys, axis=1)   # [T, 1, N, 2]
        vel_i = np.expand_dims(vel_phys, axis=2)   # [T, N, 1, 2]
        vel_j = np.expand_dims(vel_phys, axis=1)   # [T, 1, N, 2]
        course_i = np.expand_dims(course_phys, axis=2) # [T, N, 1]

        # 4. 计算相对量
        # 相对位置 (j 相对于 i)
        delta_p = pos_j - pos_i # [T, N, N, 2] (dx, dy)
        dist = np.linalg.norm(delta_p, axis=-1) # [T, N, N]
        
        # 相对速度
        delta_v = vel_j - vel_i # [T, N, N, 2] (dvx, dvy)
        rel_speed_sq = np.sum(np.square(delta_v), axis=-1) # |dv|^2
        
        # 5. 计算 TCPA (Time to Closest Point of Approach)
        #    TCPA = - (dp . dv) / |dv|^2
        #    如果 |dv|~0, TCPA 无意义 (设为大值)
        dot_p_v = np.sum(delta_p * delta_v, axis=-1)
        tcpa = -dot_p_v / (rel_speed_sq + 1e-9)
        
        # 6. 计算 DCPA (Distance at CPA)
        #    Pos_cpa = Pos_rel + Vel_rel * TCPA
        p_cpa = delta_p + delta_v * np.expand_dims(tcpa, axis=-1)
        dcpa = np.linalg.norm(p_cpa, axis=-1)

        # 7. 计算相对方位角 (Bearing) beta_ij
        #    beta = atan2(dy, dx) - course_i
        #    atan2 返回 (-pi, pi) 数学角 (从东逆时针)
        #    我们需要把 course_i 也转成数学角吗？是的。
        #    或者更简单：直接算 "视线角" 和 "船头向" 的差
        
        # 向量角度 (数学系, 逆时针)
        angle_vec = np.rad2deg(np.arctan2(delta_p[..., 1], delta_p[..., 0]))
        # 船首向 (航海系 -> 数学系)
        heading_math = 90 - course_i
        
        # 相对方位 (角度差)
        beta = angle_vec - heading_math
        # 归一化到 [-180, 180]
        beta = (beta + 180) % 360 - 180
        
        # 8. 定义相遇类型掩码 (Semantic Masks)
        #    规则参考自您的图片
        
        # A. 基础过滤器: 有碰撞风险的 (TCPA > 0 且 DCPA < D0)
        # risk_mask = (tcpa > 0) & (tcpa < self.tcpa_threshold) & (dcpa < self.dcpa_threshold)
        
        # B. 语义分类 (根据 beta)
        # 1. Head-on (头碰): |beta| < 15
        mask_head_on = (np.abs(beta) < 15)
        
        # 2. Starboard Crossing (右舷交叉): beta in (15, 112.5)
        #    在右舷意味着他船在我的右边，我有让路义务 (最重要)
        #    注意: atan2/坐标系方向要对。假设数学系下减去船首向后，
        #    负角度是在右边 (顺时针)，正角度在左边。
        #    让我们修正逻辑：beta = Target - Heading. 
        #    如果 Target 在 Heading 右边 (顺时针)，beta 应该是负的 (例如 -90)。
        #    按照您的图片: 右舷是 (15, 112.5)。这可能采用了 "右舷为正" 的定义。
        #    我们暂且严格遵守您提供的图片定义：
        mask_cross_starboard = (beta > 15) & (beta <= 112.5)
        
        # 3. Port Crossing (左舷交叉): beta in (-112.5, -15)
        mask_cross_port = (beta >= -112.5) & (beta < -15)
        
        # 4. Overtaking (追越): |beta| > 112.5
        #    并且本船速度更快 (image: "且本船更快")
        #    speed_phys [T, N] -> speed_i [T, N, 1], speed_j [T, 1, N]
        # speed_i = np.expand_dims(speed_phys, axis=2)
        # speed_j = np.expand_dims(speed_phys, axis=1)
        # mask_overtaking = (np.abs(beta) > 112.5) & (speed_i > speed_j)
        mask_overtaking = (np.abs(beta) > 112.5) 
        
        # 9. 构建 4 通道矩阵
        #    权重可以沿用之前的 "社会力权重" (dist_term * vel_term)
        #    或者简单用 1.0 / (dist + eps)
        
        # 计算基础权重 (社会力公式)
        vel_term = np.exp(-rel_speed_sq / (2 * self.social_sigma**2))
        dist_term = 1.0 / (dist + 1e-9)
        base_weight = dist_term * vel_term
        
        # 应用风险过滤
        # base_weight = base_weight * risk_mask.astype(float)
        
        # 初始化 4 通道 [T, N, N, 4]
        A_semantic = np.zeros((T, N, N, 4), dtype=np.float32)
        
        # 填入通道
        A_semantic[..., 0] = base_weight * mask_head_on
        A_semantic[..., 1] = base_weight * mask_cross_starboard
        A_semantic[..., 2] = base_weight * mask_cross_port
        A_semantic[..., 3] = base_weight * mask_overtaking
        
        # 10. 处理 "幽灵船" 掩码 (必须做!)
        # if mask is not None:
        #     if mask.dtype != bool: mask = mask.astype(bool)
        #     # [T, N, 1, 1] & [T, 1, N, 1]
        #     mask_i = mask[:, :, None, None]
        #     mask_j = mask[:, None, :, None]
        #     combined_mask = mask_i & mask_j # [T, N, N, 1]
        #     A_semantic = A_semantic * combined_mask
        # 掩码幽灵船，会导致矩阵过于稀疏，效果不好，暂时不做。
        
        # 11. 对角线清零
        diag_idx = np.arange(N)
        A_semantic[:, diag_idx, diag_idx, :] = 0.0
        
        return A_semantic

    def _build_semantic_social_fusion_matrix(self, x_data, mask=None):
        """
        构建 "语义" 社会力矩阵 (基于 COLREGs 规则)
        [完整版 - V5.3] 5通道策略: 背景层(0) + 语义层(1-4)
        
        Args:
            x_data: [T, N, D]
            mask: [T, N]
            
        Returns:
            np.ndarray: [T, N, N, 5]
            通道 0: Background (Social Force, Unmasked)
            通道 1: Head-on (Masked)
            通道 2: Starboard Crossing (Masked)
            通道 3: Port Crossing (Masked)
            通道 4: Overtaking (Masked)
        """
        T, N, D = x_data.shape
        
        # 1. 反归一化 (必须在物理空间计算几何关系)
        #    假设前4维是 [lat, lon, speed, course]
        x_phys = x_data * self.scaler_params['mean'][:D] + self.scaler_params['std'][:D]
        
        # 提取物理量
        pos_phys = x_phys[..., :2]      # [T, N, 2] (x, y)
        
        speed_phys = x_phys[..., 2]     # [T, N]
        course_phys = x_phys[..., 3]    # [T, N] (度)

        # 2. 计算速度向量 (vx, vy)
        #    假设 Course 是以北为0，顺时针 (0~360)
        #    转为数学弧度: math_angle = 90 - course
        course_rad = np.deg2rad(90 - course_phys) 
        vx = speed_phys * np.cos(course_rad)
        vy = speed_phys * np.sin(course_rad)
        vel_phys = np.stack([vx, vy], axis=-1) # [T, N, 2]

        # 3. 准备广播维度
        # i: 本船, j: 他船
        pos_i = np.expand_dims(pos_phys, axis=2)   # [T, N, 1, 2]
        pos_j = np.expand_dims(pos_phys, axis=1)   # [T, 1, N, 2]
        vel_i = np.expand_dims(vel_phys, axis=2)   # [T, N, 1, 2]
        vel_j = np.expand_dims(vel_phys, axis=1)   # [T, 1, N, 2]
        course_i = np.expand_dims(course_phys, axis=2) # [T, N, 1] (这里保持 3D)

        # 4. 计算相对量
        # 相对位置 (j 相对于 i)
        delta_p = pos_j - pos_i # [T, N, N, 2] (dx, dy)
        dist = np.linalg.norm(delta_p, axis=-1) # [T, N, N]
        
        # 相对速度
        delta_v = vel_j - vel_i # [T, N, N, 2] (dvx, dvy)
        rel_speed_sq = np.sum(np.square(delta_v), axis=-1) # |dv|^2
        
        # 5. 计算相对方位角 (Bearing) beta_ij
        
        # 向量角度 (数学系, 逆时针)
        # delta_p: [T, N, N, 2] -> angle_vec: [T, N, N]
        angle_vec = np.rad2deg(np.arctan2(delta_p[..., 1], delta_p[..., 0]))
        
        # 船首向 (航海系 -> 数学系)
        # course_i: [T, N, 1] -> heading_math: [T, N, 1]
        heading_math = 90 - course_i
        
        # 相对方位 (角度差)
        # [修正方向]: 正数=右舷 (Starboard), 负数=左舷 (Port)
        # 航向(90) - 目标方位(0) = +90 (正右方)
        beta = heading_math - angle_vec 
        
        # 归一化到 [-180, 180]
        beta = (beta + 180) % 360 - 180
        
        # 6. 定义相遇类型掩码 (Semantic Masks)
        
        # A. Head-on (头碰): [-15, 15]
        mask_head_on = (np.abs(beta) <= 15)
        
        # B. Starboard Crossing (右舷交叉): (15, 112.5]
        #    正数代表右舷
        mask_cross_starboard = (beta > 15) & (beta <= 112.5)
        
        # C. Port Crossing (左舷交叉): [-112.5, -15)
        #    负数代表左舷
        mask_cross_port = (beta >= -112.5) & (beta < -15)
        
        # D. Overtaking (追越/船尾): 绝对值 > 112.5
        mask_overtaking = (np.abs(beta) > 112.5)

        # 7. 计算基础权重 (社会力公式)
        #    权重对于所有通道都是通用的
        vel_term = np.exp(-rel_speed_sq / (2 * self.social_sigma**2))
        dist_term = 1.0 / (dist + 1e-9)
        base_weight = dist_term * vel_term

        # ==========================================
        # 8. 构建 5 通道矩阵 (核心逻辑)
        # ==========================================
        
        # --- Channel 0: 背景锚点层 (不带掩码) ---
        # 这一层保留所有交互 (包括与幽灵船的)，提供全局位置感
        A_anchor = base_weight.copy()
        A_anchor = np.expand_dims(A_anchor, axis=-1) # [T, N, N, 1]
        
        # --- Channel 1-4: 纯净语义层 (带掩码) ---
        # 这一层只保留真实船之间的规则交互
        A_rules = np.zeros((T, N, N, 4), dtype=np.float32)
        
        A_rules[..., 0] = base_weight * mask_head_on
        A_rules[..., 1] = base_weight * mask_cross_starboard
        A_rules[..., 2] = base_weight * mask_cross_port
        A_rules[..., 3] = base_weight * mask_overtaking
        
        # [关键] 对规则层应用掩码 (清零幽灵船)
        if mask is not None:
            if mask.dtype != bool: mask = mask.astype(bool)
            # [T, N, 1, 1] & [T, 1, N, 1] -> [T, N, N, 1]
            mask_i = mask[:, :, None, None]
            mask_j = mask[:, None, :, None]
            combined_mask = mask_i & mask_j 
            
            # 只 mask 规则层
            A_rules = A_rules * combined_mask

        # --- 拼接 -> 5 Channels ---
        # [T, N, N, 1] + [T, N, N, 4] -> [T, N, N, 5]
        A_final = np.concatenate([A_anchor, A_rules], axis=-1)
        A_final = A_final.astype(np.float32)
        # 9. 对角线清零 (对所有通道)
        #    船 i 对自己的力永远为 0
        diag_idx = np.arange(N)
        A_final[:, diag_idx, diag_idx, :] = 0.0
        
        return A_final
        
    def _build_edge_features(self, x_data, mask=None):
        """
        构建原始边特征 (Edge Features) - 配合 Edge-Conditioned GNN 使用
        返回: [T, N, N, 4] -> (1/d, v_rel, cos, sin)
        """
        T, N, D = x_data.shape
        
        # 1. 反归一化
        x_phys = x_data * self.scaler_params['mean'][:D] + self.scaler_params['std'][:D]
        pos_phys = x_phys[..., :2]      
        speed_phys = x_phys[..., 2]     
        course_phys = x_phys[..., 3]    

        # 2. 基础物理量准备
        course_rad = np.deg2rad(90 - course_phys) 
        vx = speed_phys * np.cos(course_rad)
        vy = speed_phys * np.sin(course_rad)
        vel_phys = np.stack([vx, vy], axis=-1)

        pos_i = np.expand_dims(pos_phys, axis=2)   
        pos_j = np.expand_dims(pos_phys, axis=1)   
        vel_i = np.expand_dims(vel_phys, axis=2)   
        vel_j = np.expand_dims(vel_phys, axis=1)   
        course_i = np.expand_dims(course_phys, axis=2) 

        # 3. 计算 4 大特征
        
        # Feature 0: 逆距离 (1/d)
        delta_p = pos_j - pos_i 
        dist = np.linalg.norm(delta_p, axis=-1) 
        inv_dist = 1.0 / (dist + 1.0) # +1.0 保持数值稳定
        
        # Feature 1: 相对速度
        delta_v = vel_j - vel_i 
        rel_speed = np.linalg.norm(delta_v, axis=-1) 
        rel_speed = np.log1p(rel_speed)
        # Feature 2 & 3: 相对方位的 Cos 和 Sin
        angle_vec = np.rad2deg(np.arctan2(delta_p[..., 1], delta_p[..., 0]))
        heading_math = 90 - course_i
        # 正数=右舷, 负数=左舷
        beta_deg = heading_math - angle_vec 
        beta_rad = np.deg2rad(beta_deg)
        
        cos_beta = np.cos(beta_rad)
        sin_beta = np.sin(beta_rad)
        
        # 4. 堆叠 -> [T, N, N, 4]
        edge_features = np.stack([inv_dist, rel_speed, cos_beta, sin_beta], axis=-1)
        
        # 5. 掩码幽灵船 (必须做，否则 MLP 会学到噪声)
        if mask is not None:
            if mask.dtype != bool: mask = mask.astype(bool)
            mask_i = mask[:, :, None, None]
            mask_j = mask[:, None, :, None]
            combined_mask = mask_i & mask_j 
            edge_features = edge_features * combined_mask
            
        return edge_features.astype(np.float32)

    def _build_lane_features(self, x_data, mask):
        """
        基于当前样本的物理经纬度 + 航道几何，构建 [T,N,2] 的 lane 特征：
        [...,0]=s_norm(0~1), [...,1]=d_signed
        x_data: [T,N,D]  当前是**归一化后的**4维特征
        mask:   [T,N]    1=有效船舶
        """
        assert self.lanes is not None, "lanes not loaded"

        T, N, D = x_data.shape
        lane_s = np.zeros((T, N), dtype=np.float32)
        lane_d = np.zeros((T, N), dtype=np.float32)

        # 先反归一化到物理空间
        x_phys = x_data.copy()
        if self.scale_type == 'standard':
            for f in range(min(D, self.num_features)):
                x_phys[..., f] = x_phys[..., f] * self.std[f] + self.mean[f]
        elif self.scale_type == 'minmax':
            for f in range(min(D, self.num_features)):
                x_phys[..., f] = x_phys[..., f] * (self.max_val[f] - self.min_val[f]) + self.min_val[f]

        # 这里假设 feature 顺序是 [lon, lat, SOG, COG]
        lon = x_phys[..., 0]
        lat = x_phys[..., 1]

        m = (mask.astype(bool) if mask is not None else np.ones((T, N), dtype=bool))

        for t in range(T):
            for n in range(N):
                if not m[t, n]:
                    continue
                p = Point(lon[t, n], lat[t, n])
                name, entry, _ = find_nearest_lane(self.lanes, lon[t, n], lat[t, n])
                if entry is None:
                    continue
                s_norm, d_signed = lane_features_for_point(p, entry)
                lane_s[t, n] = s_norm
                lane_d[t, n] = d_signed

        feats = np.stack([lane_s, lane_d], axis=-1)   # [T,N,2]
        return feats



def ship_collate_fn(batch):
    seq_x_list, seq_y_list, mask_x_list, mask_y_list, count_list, id_list = zip(*batch)
    seq_x = torch.from_numpy(np.array(seq_x_list)).float()
    seq_y = torch.from_numpy(np.array(seq_y_list)).float()
    mask_x = torch.from_numpy(np.array(mask_x_list)).bool()   # <- 改成 bool
    mask_y = torch.from_numpy(np.array(mask_y_list)).bool()   # <- 改成 bool
    ship_count = torch.from_numpy(np.array(count_list)).long()
    global_id  = torch.from_numpy(np.array(id_list)).long()
    return seq_x, seq_y, mask_x, mask_y, ship_count, global_id

class ShipTrajectoryDataLoader:
    """
    数据加载器工厂类
    简化数据加载器的创建
    """
    
    def __init__(self, args):
        self.args = args
    
    def get_data_loader(self, flag):
        """
        获取数据加载器
        
        Args:
            flag: 'train', 'val', 'test'
        
        Returns:
            dataset, dataloader
        """
        # 数据文件路径
        data_dict = {
            'train': 'train.npz',
            'val': 'val.npz',
            'test': 'test.npz'
        }
        
        # 是否打乱数据
        shuffle_flag = (flag == 'train')
        drop_last = (flag == 'train')
        
        # 创建数据集
        dataset = ShipTrajectoryDataset(
            root_path=self.args.root_path,
            data_path=data_dict[flag],
            flag=flag,
            size=[self.args.seq_len, self.args.pred_len],
            num_ships=self.args.num_ships,
            num_features=self.args.num_features,
            scale=self.args.scale,
            scale_type=self.args.scale_type,
            predict_position_only=self.args.predict_position_only
        )
        
        # 保存训练集的归一化参数
        if flag == 'train':
            dataset.save_scaler_params()
        
        # 创建数据加载器
        dataloader = DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            shuffle=shuffle_flag,
            num_workers=self.args.num_workers,
            drop_last=drop_last,
            pin_memory=True,
            # collate_fn=ship_collate_fn
        )
        
        return dataset, dataloader


# ==================== 自定义collate函数（可选）====================

def ship_collate_fn(batch):
    """
    自定义collate函数，用于DataLoader
    可以在这里做一些批处理的特殊操作
    
    Args:
        batch: list of tuples (seq_x, seq_y, seq_x_mask, seq_y_mask, ship_count, global_id)
    
    Returns:
        batched tensors
    """
    seq_x_list, seq_y_list, mask_x_list, mask_y_list, count_list, id_list = zip(*batch)
    
    # 转换为tensor
    seq_x = torch.FloatTensor(np.array(seq_x_list))
    seq_y = torch.FloatTensor(np.array(seq_y_list))
    mask_x = torch.FloatTensor(np.array(mask_x_list))
    mask_y = torch.FloatTensor(np.array(mask_y_list))
    ship_count = torch.LongTensor(np.array(count_list))
    global_id = torch.LongTensor(np.array(id_list))
    
    return seq_x, seq_y, mask_x, mask_y, ship_count, global_id


# ==================== 测试代码 ====================

if __name__ == '__main__':
    """测试数据加载"""
    
    class Args:
        root_path = './data/'
        seq_len = 8
        pred_len = 12
        num_ships = 17
        num_features = 4
        batch_size = 32
        num_workers = 4
        scale = True
        scale_type = 'standard'  # 'standard' or 'minmax'
        predict_position_only = True
    
    args = Args()
    
    # 创建数据加载器
    loader_factory = ShipTrajectoryDataLoader(args)
    
    # 测试训练集
    print('=' * 50)
    print('Testing train dataset...')
    print('=' * 50)
    train_dataset, train_loader = loader_factory.get_data_loader('train')
    
    # 获取一个批次
    for batch_x, batch_y, mask_x, mask_y, ship_count, global_id in train_loader:
        print(f'\nBatch shapes:')
        print(f'  batch_x: {batch_x.shape}')  # [B, T_in, N, D]
        print(f'  batch_y: {batch_y.shape}')  # [B, T_out, N, D]
        print(f'  mask_x: {mask_x.shape}')   # [B, T_in, N]
        print(f'  mask_y: {mask_y.shape}')   # [B, T_out, N]
        print(f'  ship_count: {ship_count.shape}')  # [B]
        print(f'  global_id: {global_id.shape}')    # [B, N]
        
        print(f'\nSample statistics:')
        print(f'  Ship counts in batch: {ship_count.numpy()}')
        print(f'  X value range: [{batch_x.min():.4f}, {batch_x.max():.4f}]')
        print(f'  y value range: [{batch_y.min():.4f}, {batch_y.max():.4f}]')
        # print(f'  X mask coverage: {mask_x.mean():.4f}')
        # print(f'  y mask coverage: {mask_y.mean():.4f}')
        
        break
    
    # 测试验证集
    print('\n' + '=' * 50)
    print('Testing val dataset...')
    print('=' * 50)
    val_dataset, val_loader = loader_factory.get_data_loader('val')
    
    # 测试测试集
    print('\n' + '=' * 50)
    print('Testing test dataset...')
    print('=' * 50)
    test_dataset, test_loader = loader_factory.get_data_loader('test')
    
    print('\n' + '=' * 50)
    print('All tests passed!')
    print('=' * 50)