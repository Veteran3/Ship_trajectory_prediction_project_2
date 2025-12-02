from shapely.geometry import LineString, Point
from shapely.ops import nearest_points
import pandas as pd
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import warnings
import time
import hashlib
import json
from multiprocessing import Pool, cpu_count
import pickle
from tqdm import tqdm  # 【新增】引入 tqdm

warnings.filterwarnings('ignore')

"""
v5.1 深度优化版 (Precision Optimized) + TQDM 可视化
核心策略：
1. 逻辑保持：严格保留 v5.1 的特征定义 (2D切线方向, s, d)。
2. 精度保证：只缓存“航道查找结果(Lookup)”，特征值在运行时基于原始坐标实时计算，避免浮点缓存误差。
3. 速度优化：多进程预计算最耗时的“查找”步骤。
4. 可视化：使用 tqdm 显示处理进度，适应密集航道点带来的长耗时。
"""

# ============================================================================
# 1. 缓存管理器 (改为缓存查找结果，而非特征值)
# ============================================================================

class LaneLookupCache:
    """
    航道查找缓存管理器
    缓存内容：每个 AIS 点对应的 [lane_name, seq, next_lanes, raw_lon, raw_lat]
    """
    
    def __init__(self, root_path, flag, data_shape, lane_table_path):
        # 使用 v5_1_opt 后缀以示区别
        self.cache_dir = os.path.join(root_path, 'lane_lookup_cache_v5_0')
        os.makedirs(self.cache_dir, exist_ok=True)
        
        self.flag = flag
        self.cache_id = self._generate_cache_id(data_shape, lane_table_path)
        
        self.cache_path = os.path.join(self.cache_dir, f'{flag}_lookup.pkl')
        self.metadata_path = os.path.join(self.cache_dir, f'{flag}_metadata.json')
    
    def _generate_cache_id(self, data_shape, lane_table_path):
        # 生成唯一指纹，确保数据源变动时缓存失效
        content = f"{data_shape}_{lane_table_path}_{os.path.getmtime(lane_table_path) if lane_table_path else 0}"
        return hashlib.md5(content.encode()).hexdigest()[:16]
    
    def is_valid(self):
        if not os.path.exists(self.metadata_path):
            return False
        try:
            with open(self.metadata_path, 'r') as f:
                metadata = json.load(f)
            return (metadata.get('cache_id') == self.cache_id and
                    metadata.get('completed', False) and
                    os.path.exists(self.cache_path))
        except:
            return False
    
    def load(self):
        print(f"Loading cache from {self.cache_path} ...")
        with open(self.cache_path, 'rb') as f:
            return pickle.load(f)
    
    def save(self, lookup_results):
        print(f"Saving cache to {self.cache_path} ...")
        with open(self.cache_path, 'wb') as f:
            pickle.dump(lookup_results, f, protocol=4)
        
        metadata = {
            'cache_id': self.cache_id,
            'completed': True,
            'timestamp': time.time()
        }
        with open(self.metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)


# ============================================================================
# 2. 静态辅助函数 (用于多进程和运行时计算)
# ============================================================================

def load_lane_table(path: str):
    ext = os.path.splitext(path)[1].lower()
    if ext in [".xlsx", ".xls"]:
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path, encoding="utf-8")
    
    df = df.rename(columns={
        df.columns[0]: "lane_name",
        df.columns[1]: "lane_dir",
        df.columns[2]: "lane_role",
        df.columns[3]: "seq",
        df.columns[4]: "lon",
        df.columns[5]: "lat",
    })
    return df

def build_lanes(df: pd.DataFrame):
    lanes = {}
    for (lane_name, lane_dir, lane_role), g in df.groupby(["lane_name", "lane_dir", "lane_role"]):
        coords = list(zip(g["lon"].values, g["lat"].values))
        if len(coords) < 2:
            continue
        line = LineString(coords)
        entry = lanes.setdefault(lane_name, {"dir": lane_dir})
        entry[lane_role] = line
    return lanes

def find_nearest_lane(lanes, lon, lat):
    """几何查找：找到最近的航道"""
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

def get_seq_for_lane_point_static(lane_name, lon, lat, lanes):
    """几何计算：找到点在航道上的序号"""
    lane_data = lanes.get(lane_name, None)
    if lane_data is None: return -1
    center_line = lane_data["center"]
    point = Point(lon, lat)
    min_distance = float('inf')
    seq = None
    # 注意：如果航道插值后点非常密集，这个循环是耗时的主要原因
    for i, coord in enumerate(center_line.coords, 1):
        distance = point.distance(Point(coord))
        if distance < min_distance:
            min_distance = distance
            seq = i
    return seq if seq is not None else -1

def get_next_lane_static(lane_name, seq, df_lanes):
    """数据表查找：获取 next_lanes 列表"""
    lane_info = df_lanes[(df_lanes['lane_name'] == lane_name) & (df_lanes['seq'] == seq)]
    if not lane_info.empty:
        next_lanes_str = lane_info['next_lane'].values[0]
        # 鲁棒的字符串解析
        if pd.isna(next_lanes_str): return []
        next_lanes = str(next_lanes_str).strip("[]").replace("'", "").replace('"', '').split(',')
        return [lane.strip() for lane in next_lanes if lane.strip()]
    return []

def next_lanes_to_onehot(next_lanes, all_lanes=["EW1", "NS1", "NS2", "EW2", "WE1", "WE2", "SN1", "SN2"]):
    onehot = np.zeros(len(all_lanes), dtype=np.float32)
    if isinstance(next_lanes, str): next_lanes = [next_lanes]
    if next_lanes is None: return onehot
    
    for lane in next_lanes:
        if lane in all_lanes:
            idx = all_lanes.index(lane)
            onehot[idx] = 1.0
    return onehot

def lane_features_for_point_v51(p: Point, lane_entry):
    """
    【v5.1 原版逻辑】计算位置特征 s, d
    """
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

def compute_dir_features_v51(p: Point, lane_entry):
    """
    【v5.1 原版逻辑】计算方向特征 (当前航道切线)
    """
    line_geom = lane_entry['center']
    # 找到最近点
    p_on = nearest_points(p, line_geom)[1]
    s = line_geom.project(p_on)
    
    # 前后差分计算切线
    ds = 1.0
    s1 = max(0.0, s - ds)
    s2 = min(line_geom.length, s + ds)
    p1 = line_geom.interpolate(s1)
    p2 = line_geom.interpolate(s2)
    
    dx = p2.x - p1.x
    dy = p2.y - p1.y
    norm = (dx**2 + dy**2) ** 0.5
    
    if norm > 1e-6:
        return dx / norm, dy / norm
    else:
        return 0.0, 0.0


# ============================================================================
# 3. 多进程工作函数 (只做查找，不计算 Float)
# ============================================================================

def _lookup_lanes_chunk(args):
    """
    Worker: 只负责找到 lane_name, seq 和 next_lanes。
    不进行任何 float 计算，避免精度损失和数据冗余。
    """
    sample_indices, lon_phys_chunk, lat_phys_chunk, mask_chunk, lanes, df_lanes = args
    
    B_chunk, T, N = lon_phys_chunk.shape
    lookup_results = []
    
    for local_idx in range(B_chunk):
        sample_result = []
        for t in range(T):
            time_result = []
            for n in range(N):
                if not mask_chunk[local_idx, t, n]:
                    time_result.append(None)
                    continue
                
                lon = lon_phys_chunk[local_idx, t, n]
                lat = lat_phys_chunk[local_idx, t, n]
                
                try:
                    # 1. 查找最近航道
                    lane_name, _, _ = find_nearest_lane(lanes, lon, lat)
                    
                    if lane_name is None:
                        time_result.append(None)
                        continue
                    
                    # 2. 获取序号 (航道插值后点变多，这里会变慢)
                    seq = get_seq_for_lane_point_static(lane_name, lon, lat, lanes)
                    
                    # 3. 查找 next_lanes (字符串列表)
                    next_lanes = get_next_lane_static(lane_name, seq, df_lanes)
                    
                    # 存储结果
                    time_result.append({
                        'lane_name': lane_name,
                        'seq': seq,
                        'next_lanes': next_lanes,
                        'lon': float(lon), 
                        'lat': float(lat)
                    })
                except:
                    time_result.append(None)
            sample_result.append(time_result)
        lookup_results.append(sample_result)
    
    return lookup_results


# ============================================================================
# 4. Dataset 类 (v5.1 Optimized + TQDM)
# ============================================================================

class ShipTrajectoryDataset(Dataset):
    
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
        lane_table_path=None,
        force_recompute=False
    ):
        self.root_path = root_path
        self.data_path = data_path
        self.flag = flag
        self.num_ships = num_ships
        self.num_features = num_features
        self.scale = scale
        self.scale_type = scale_type
        self.predict_position_only = predict_position_only
        self.social_sigma = 1.0
        self.force_recompute = force_recompute
        
        # 加载航道表
        self.lanes = None
        self.df_lanes = None
        self.lane_table_path = lane_table_path
        
        if lane_table_path is not None and os.path.exists(lane_table_path):
            self.df_lanes = load_lane_table(lane_table_path)
            self.lanes = build_lanes(self.df_lanes)
        
        # 序列长度
        if size is None:
            self.seq_len = 8
            self.pred_len = 12
        else:
            self.seq_len = size[0]
            self.pred_len = size[1]
        
        self.lane_lookup_cache = None
        
        # 加载数据
        self.__read_data__()
    
    def __read_data__(self):
        full_path = os.path.join(self.root_path, self.data_path)
        print(f'\n{"="*70}')
        print(f'Loading {self.flag} dataset from: {full_path}')
        print(f'{"="*70}')
        
        data = np.load(full_path)
        self.data_x = data['X']
        self.data_y = data['y']
        self.mask_x = data['X_mask']
        self.mask_y = data['y_mask']
        self.ship_counts = data['ship_counts']
        self.global_ids = data['global_ids']
        
        B, T, N, D = self.data_x.shape
        print(f'  Samples: {B:,}')
        print(f'  Shape: X{self.data_x.shape}, y{self.data_y.shape}')
        print(f'  Memory: ~{self.data_x.nbytes / 1024**3:.2f} GB')
        
        # 归一化
        if self.scale:
            self.__normalize_data__()
        
        # 预计算航道特征
        if self.lanes is not None:
            self.__precompute_lane_lookup__()
        
        print(f'{"="*70}\n')
    
    def __precompute_lane_lookup__(self):
        """预计算航道归属，建立索引缓存 (带进度条)"""
        B, T, N, D = self.data_x.shape
        
        cache = LaneLookupCache(
            self.root_path, self.flag, (B, T, N, D), self.lane_table_path
        )
        
        # 尝试加载缓存
        if not self.force_recompute and cache.is_valid():
            print(f"[{self.flag}] Loading valid lane lookup cache...")
            self.lane_lookup_cache = cache.load()
            return
        
        print(f"[{self.flag}] Computing lane lookup for {B:,} samples (multiprocessing)...")
        print(f"          Note: Using {min(cpu_count(), 8)} cores. Dense lanes may take longer.")
        
        # 反归一化获取物理坐标用于查找
        lon_phys = self.data_x[:, :, :, 0] * self.std[0] + self.mean[0]
        lat_phys = self.data_x[:, :, :, 1] * self.std[1] + self.mean[1]
        
        # 多进程分块
        num_processes = min(cpu_count(), 8)
        chunk_size = max(100, B // (num_processes * 4))
        sample_chunks = [list(range(i, min(i + chunk_size, B))) for i in range(0, B, chunk_size)]
        
        chunk_args = []
        for chunk in sample_chunks:
            chunk_args.append((
                chunk, lon_phys[chunk], lat_phys[chunk], self.mask_x[chunk], 
                self.lanes, self.df_lanes
            ))
        
        all_results = []
        start_time = time.time()
        
        # 【新增】使用 tqdm 显示进度
        with Pool(processes=num_processes) as pool:
            # imap 按顺序返回结果
            result_iter = pool.imap(_lookup_lanes_chunk, chunk_args)
            
            # 使用 tqdm 包装迭代器
            for chunk_result in tqdm(result_iter, total=len(sample_chunks), desc="  Lane Lookup", unit="chunk"):
                all_results.extend(chunk_result)
                    
        total_time = time.time() - start_time
        print(f"  ✓ Lookup completed in {total_time:.1f}s")
        
        cache.save(all_results)
        self.lane_lookup_cache = all_results

    def __getitem__(self, index):
        """
        获取单个样本。
        在此处实时计算 v5.1 的几何特征，保证精度。
        """
        seq_x = self.data_x[index].copy()
        seq_y = self.data_y[index].copy()
        seq_x_mask = self.mask_x[index].copy()
        seq_y_mask = self.mask_y[index].copy()
        ship_count = self.ship_counts[index]
        global_id = self.global_ids[index].copy()
        
        # 1. 计算 Social Matrix (保持不变)
        A_social = self._build_semantic_social_fusion_matrix(seq_x)
        edge_features = self._build_edge_features(seq_x, mask=seq_x_mask)
        
        # 2. 计算航道特征 (v5.1 逻辑, 实时计算)
        if self.lanes is not None and self.lane_lookup_cache is not None:
            lane_feats, lane_dir_feats, next_lanes_feats = self._compute_lane_features_v51_runtime(
                index, seq_x_mask
            )
        else:
            T, N = seq_x.shape[0], seq_x.shape[1]
            lane_feats = np.zeros((T, N, 2), dtype=np.float32)
            lane_dir_feats = np.zeros((T, N, 2), dtype=np.float32)
            next_lanes_feats = np.zeros((T, N, 8), dtype=np.float32)
        
        # 3. 特征转换与拼接
        seq_x = self._transform_features(seq_x, seq_x_mask)
        seq_y = self._transform_features(seq_y, seq_y_mask)
        
        seq_x_final = np.concatenate([seq_x, lane_feats, next_lanes_feats, lane_dir_feats], axis=-1)
        
        return seq_x_final, seq_y, seq_x_mask, seq_y_mask, ship_count, global_id, A_social, edge_features

    def _compute_lane_features_v51_runtime(self, index, seq_x_mask):
        """
        【核心】运行时计算 v5.1 特征
        使用缓存的 lookup 结果，但实时执行几何投影计算。
        """
        lookup_sample = self.lane_lookup_cache[index]
        T, N = seq_x_mask.shape
        
        lane_feats = np.zeros((T, N, 2), dtype=np.float32)     # [s, d]
        lane_dir_feats = np.zeros((T, N, 2), dtype=np.float32) # [cos, sin] (2D v5.1)
        next_lanes_feats = np.zeros((T, N, 8), dtype=np.float32)
        
        for t in range(T):
            for n in range(N):
                if not seq_x_mask[t, n]:
                    continue
                
                info = lookup_sample[t][n]
                if info is None:
                    continue
                
                try:
                    # 取出预查找的数据
                    lane_name = info['lane_name']
                    lon = info['lon'] # 原始坐标，精度高
                    lat = info['lat']
                    next_lanes = info['next_lanes']
                    
                    lane_entry = self.lanes[lane_name]
                    p = Point(lon, lat)
                    
                    # 1. 计算 s, d (严格遵守 v5.1 逻辑)
                    s_norm, d_signed = lane_features_for_point_v51(p, lane_entry)
                    lane_feats[t, n, 0] = s_norm
                    lane_feats[t, n, 1] = d_signed
                    
                    # 2. 计算方向 (严格遵守 v5.1 逻辑：当前航道切线)
                    dx, dy = compute_dir_features_v51(p, lane_entry)
                    lane_dir_feats[t, n, 0] = dx
                    lane_dir_feats[t, n, 1] = dy
                    
                    # 3. Next lanes One-hot
                    next_lanes_onehot = next_lanes_to_onehot(next_lanes)
                    next_lanes_feats[t, n, :] = next_lanes_onehot
                    
                except Exception as e:
                    pass
                    
        return lane_feats, lane_dir_feats, next_lanes_feats

    # ==========================
    # 以下为标准辅助函数 (保持不变)
    # ==========================
    
    def __normalize_data__(self):
        if self.flag == 'train':
            self.mean = np.zeros(self.num_features)
            self.std = np.zeros(self.num_features)
            self.min_val = np.zeros(self.num_features)
            self.max_val = np.zeros(self.num_features)
            
            for feat_idx in range(self.num_features):
                valid_data = self.data_x[:, :, :, feat_idx][self.mask_x == 1]
                if self.scale_type == 'standard':
                    self.mean[feat_idx] = valid_data.mean()
                    self.std[feat_idx] = valid_data.std()
                elif self.scale_type == 'minmax':
                    self.min_val[feat_idx] = valid_data.min()
                    self.max_val[feat_idx] = valid_data.max()
            
            if self.scale_type == 'standard':
                self.std = np.where(self.std < 1e-8, 1.0, self.std)
            elif self.scale_type == 'minmax':
                span = self.max_val - self.min_val
                span = np.where(span < 1e-8, 1.0, span)
                self.max_val = self.min_val + span
            
            self.scaler_params = {'mean': self.mean, 'std': self.std, 'min_val': self.min_val, 'max_val': self.max_val, 'scale_type': self.scale_type}
            np.save(os.path.join(self.root_path, 'scaler_params.npy'), self.scaler_params)
            self._apply_normalization(self.data_x, self.mask_x)
            self._apply_normalization(self.data_y, self.mask_y)
        else:
            scaler_path = os.path.join(self.root_path, 'scaler_params.npy')
            if os.path.exists(scaler_path):
                self.scaler_params = np.load(scaler_path, allow_pickle=True).item()
                self.mean = self.scaler_params['mean']
                self.std = self.scaler_params['std']
                self.min_val = self.scaler_params['min_val']
                self.max_val = self.scaler_params['max_val']
                self.scale_type = self.scaler_params['scale_type']
                self._apply_normalization(self.data_x, self.mask_x)
                self._apply_normalization(self.data_y, self.mask_y)

    def _apply_normalization(self, data, mask):
        B, T, N, Dp = data.shape
        F = min(self.num_features, Dp)
        m = mask.astype(np.bool_)
        if self.scale_type == 'standard':
            for f in range(F):
                data[..., f] = np.where(m, (data[..., f] - self.mean[f]) / (self.std[f] + 1e-8), data[..., f])
        elif self.scale_type == 'minmax':
            for f in range(F):
                data[..., f] = np.where(m, (data[..., f] - self.min_val[f]) / (self.max_val[f] - self.min_val[f] + 1e-8), data[..., f])

    def _transform_features(self, x, mask):
        feats_others = x[..., :3]
        cog_scaled = x[..., 3]
        if self.scale_type == 'standard':
            cog_phys = cog_scaled * self.std[3] + self.mean[3]
        elif self.scale_type == 'minmax':
            cog_phys = cog_scaled * (self.max_val[3] - self.min_val[3]) + self.min_val[3]
        cog_rad = np.deg2rad(cog_phys)
        cog_cos = np.cos(cog_rad)
        cog_sin = np.sin(cog_rad)
        if mask.dtype != bool: mask = mask.astype(bool)
        new_feats = np.concatenate([feats_others, cog_cos[..., np.newaxis], cog_sin[..., np.newaxis]], axis=-1)
        new_feats = new_feats * mask[..., np.newaxis]
        return new_feats.astype(np.float32)
    
    def _build_semantic_social_fusion_matrix(self, x_data, mask=None):
        """语义社会力矩阵构建 (保持逻辑一致)"""
        T, N, D = x_data.shape
        x_phys = x_data * self.scaler_params['mean'][:D] + self.scaler_params['std'][:D]
        pos_phys = x_phys[..., :2]
        speed_phys = x_phys[..., 2]
        course_phys = x_phys[..., 3]
        course_rad = np.deg2rad(90 - course_phys)
        vx = speed_phys * np.cos(course_rad)
        vy = speed_phys * np.sin(course_rad)
        vel_phys = np.stack([vx, vy], axis=-1)
        
        pos_i = np.expand_dims(pos_phys, axis=2)
        pos_j = np.expand_dims(pos_phys, axis=1)
        vel_i = np.expand_dims(vel_phys, axis=2)
        vel_j = np.expand_dims(vel_phys, axis=1)
        course_i = np.expand_dims(course_phys, axis=2)
        
        delta_p = pos_j - pos_i
        dist = np.linalg.norm(delta_p, axis=-1)
        delta_v = vel_j - vel_i
        rel_speed_sq = np.sum(np.square(delta_v), axis=-1)
        
        angle_vec = np.rad2deg(np.arctan2(delta_p[..., 1], delta_p[..., 0]))
        heading_math = 90 - course_i
        beta = heading_math - angle_vec
        beta = (beta + 180) % 360 - 180
        
        mask_head_on = (np.abs(beta) <= 15)
        mask_cross_starboard = (beta > 15) & (beta <= 112.5)
        mask_cross_port = (beta >= -112.5) & (beta < -15)
        mask_overtaking = (np.abs(beta) > 112.5)
        
        vel_term = np.exp(-rel_speed_sq / (2 * self.social_sigma**2))
        dist_term = 1.0 / (dist + 1e-9)
        base_weight = dist_term * vel_term
        
        A_anchor = np.expand_dims(base_weight.copy(), axis=-1)
        A_rules = np.zeros((T, N, N, 4), dtype=np.float32)
        A_rules[..., 0] = base_weight * mask_head_on
        A_rules[..., 1] = base_weight * mask_cross_starboard
        A_rules[..., 2] = base_weight * mask_cross_port
        A_rules[..., 3] = base_weight * mask_overtaking
        
        if mask is not None:
            if mask.dtype != bool: mask = mask.astype(bool)
            combined_mask = mask[:, :, None, None] & mask[:, None, :, None]
            A_rules = A_rules * combined_mask
            
        A_final = np.concatenate([A_anchor, A_rules], axis=-1).astype(np.float32)
        diag_idx = np.arange(N)
        A_final[:, diag_idx, diag_idx, :] = 0.0
        return A_final

    def _build_edge_features(self, x_data, mask=None):
        T, N, D = x_data.shape
        x_phys = x_data * self.scaler_params['mean'][:D] + self.scaler_params['std'][:D]
        pos_phys = x_phys[..., :2]
        speed_phys = x_phys[..., 2]
        course_phys = x_phys[..., 3]
        
        course_rad = np.deg2rad(90 - course_phys)
        vx = speed_phys * np.cos(course_rad)
        vy = speed_phys * np.sin(course_rad)
        vel_phys = np.stack([vx, vy], axis=-1)
        
        pos_i = np.expand_dims(pos_phys, axis=2)
        pos_j = np.expand_dims(pos_phys, axis=1)
        vel_i = np.expand_dims(vel_phys, axis=2)
        vel_j = np.expand_dims(vel_phys, axis=1)
        course_i = np.expand_dims(course_phys, axis=2)
        
        delta_p = pos_j - pos_i
        dist = np.linalg.norm(delta_p, axis=-1)
        inv_dist = 1.0 / (dist + 1.0)
        
        delta_v = vel_j - vel_i
        rel_speed = np.linalg.norm(delta_v, axis=-1)
        rel_speed = np.log1p(rel_speed)
        
        angle_vec = np.rad2deg(np.arctan2(delta_p[..., 1], delta_p[..., 0]))
        heading_math = 90 - course_i
        beta_deg = heading_math - angle_vec
        beta_rad = np.deg2rad(beta_deg)
        cos_beta = np.cos(beta_rad)
        sin_beta = np.sin(beta_rad)
        
        edge_features = np.stack([inv_dist, rel_speed, cos_beta, sin_beta], axis=-1)
        if mask is not None:
            if mask.dtype != bool: mask = mask.astype(bool)
            combined_mask = mask[:, :, None, None] & mask[:, None, :, None]
            edge_features = edge_features * combined_mask
        return edge_features.astype(np.float32)
    
    def inverse_transform(self, data):
        if not self.scale: return data
        data_copy = data.copy()
        for f in range(data.shape[-1]):
            if self.scale_type == 'standard':
                data_copy[..., f] = data_copy[..., f] * self.std[f] + self.mean[f]
            elif self.scale_type == 'minmax':
                data_copy[..., f] = data_copy[..., f] * (self.max_val[f] - self.min_val[f]) + self.min_val[f]
        return data_copy
    
    def save_scaler_params(self, save_path=None):
        if self.flag == 'train' and self.scale:
            if save_path is None: save_path = os.path.join(self.root_path, 'scaler_params.npy')
            np.save(save_path, self.scaler_params)

    def __len__(self):
        return len(self.data_x)

# ============================================================================
# 5. Loader Factory
# ============================================================================

class ShipTrajectoryDataLoader:
    def __init__(self, args):
        self.args = args
    
    def get_data_loader(self, flag):
        data_dict = {'train': 'train.npz', 'val': 'val.npz', 'test': 'test.npz'}
        shuffle_flag = (flag == 'train')
        drop_last = (flag == 'train')
        
        dataset = ShipTrajectoryDataset(
            root_path=self.args.root_path,
            data_path=data_dict[flag],
            flag=flag,
            size=[self.args.seq_len, self.args.pred_len],
            num_ships=self.args.num_ships,
            num_features=self.args.num_features,
            scale=self.args.scale,
            scale_type=self.args.scale_type,
            predict_position_only=self.args.predict_position_only,
            lane_table_path=getattr(self.args, 'lane_table_path', None),
            force_recompute=getattr(self.args, 'force_recompute', False)
        )
        
        if flag == 'train':
            dataset.save_scaler_params()
        
        dataloader = DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            shuffle=shuffle_flag,
            num_workers=self.args.num_workers,
            drop_last=drop_last,
            pin_memory=True,
        )
        return dataset, dataloader

if __name__ == '__main__':
    # 测试代码区
    class Args:
        root_path = './data/30s/'
        seq_len = 8
        pred_len = 12
        num_ships = 17
        num_features = 4
        batch_size = 32
        num_workers = 4
        scale = True
        scale_type = 'standard'
        predict_position_only = True
        lane_table_path = './data/lane_table_with_next_lane.csv'
        force_recompute = False
    
    args = Args()
    print("Testing Optimized v5.1 Loader with TQDM...")
    loader = ShipTrajectoryDataLoader(args)
    dataset, dataloader = loader.get_data_loader('train')
    
    start = time.time()
    for i, batch in enumerate(dataloader):
        if i >= 5: break
    print(f"5 batches processed in {time.time()-start:.2f}s")