import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from shapely.geometry import Point, LineString
from shapely.ops import nearest_points
import math
import ast  # 用于解析 CSV 中的列表字符串 "['NS1', 'EW1']"

class LaneDataset(Dataset):
    def __init__(self, data_path, lane_csv_path, t_in=40, t_out=60, 
                 vals_to_norm=None, max_branches=2):
        """
        :param max_branches: 考虑的最大分支数 (默认2)
        """
        self.t_in = t_in
        self.t_out = t_out
        self.max_branches = max_branches
        
        # 1. 读取数据
        with open(data_path, 'rb') as f:
            raw_data = pickle.load(f)
        self.seq_x = raw_data['seq_x']
        self.seq_y = raw_data['seq_y']
        self.seq_x_mask = raw_data['seq_x_mask']
        self.seq_y_mask = raw_data['seq_y_mask']
        
        # 归一化参数
        if vals_to_norm is None:
            # 如果没有传，这就只是个 placeholder，实际应该传入
            self.vals_to_norm = {
                'min_vals': np.zeros(6), 
                'max_vals': np.ones(6)
            }
        else:
            self.vals_to_norm = vals_to_norm
            
        # 2. 读取并处理航道信息
        self.df_lanes = pd.read_csv(lane_csv_path)
        # 解析 next_lane 列: "['NS1']" -> ['NS1']
        # 注意：fillna("[]") 防止空值报错
        self.df_lanes['next_lane'] = self.df_lanes['next_lane'].fillna("[]").apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else x
        )
        
        self.lanes = self.build_lanes(self.df_lanes)
        
        # 3. 【新功能】预计算航道连接点 (Connectivity Lookup)
        #    解决 "入口点在哪" 的几何问题
        self.connectivity = self._build_connectivity_lookup()

    def build_lanes(self, df):
        """
        将 DataFrame 转为 {lane_name: {'center': LineString, ...}}
        """
        lanes = {}
        grouped = df.groupby(['lane_name', 'lane_dir', 'lane_role'])
        
        for (name, direct, role), group in grouped:
            if role != 'center': continue # 只关心中心线
            
            # 排序：根据 lane_dir 决定点的顺序
            if direct in ['WE', 'SN']: # 假设 WE 是经度增，SN 是纬度增 (按你的逻辑调整)
                 group = group.sort_values(by=['lon']) if direct == 'WE' else group.sort_values(by=['lat'])
            else: # EW 经度降，NS 纬度降
                 group = group.sort_values(by=['lon'], ascending=False) if direct == 'EW' else group.sort_values(by=['lat'], ascending=False)
            
            coords = group[['lon', 'lat']].values
            if len(coords) > 1:
                line = LineString(coords)
                if name not in lanes:
                    lanes[name] = {}
                lanes[name]['center'] = line
        return lanes

    def _build_connectivity_lookup(self):
        """
        【核心修改】
        预计算任意两条航道 (Curr, Next) 之间的最佳几何连接点。
        使用 shapely.nearest_points 解决 "汇入/交叉/首尾相接" 等各种情况。
        """
        connectivity = {}
        
        # 1. 提取所有出现的 (curr, next) 对
        unique_pairs = set()
        for _, row in self.df_lanes.iterrows():
            curr_name = row['lane_name']
            next_list = row['next_lane']
            if isinstance(next_list, list):
                for next_name in next_list:
                    unique_pairs.add((curr_name, next_name))
        
        # 2. 计算连接几何
        for curr_name, next_name in unique_pairs:
            # 同样航道(Keep Lane)不需要静态点，动态计算即可
            if curr_name == next_name:
                continue
            
            if curr_name not in self.lanes or next_name not in self.lanes:
                continue
                
            line_curr = self.lanes[curr_name]['center']
            line_next = self.lanes[next_name]['center']
            
            # 找到两条线之间最近的点 (通常是交叉点或 endpoint)
            # p_on_curr: 当前航道上离下一条最近的点
            # p_entry:   下一条航道上的入口点
            _, p_entry = nearest_points(line_curr, line_next)
            
            # 计算 p_entry 处的切线方向 (用于 Heading Diff)
            # 这里的逻辑是：取 next_lane 上 p_entry 稍微往后一点的点做差分
            dist_proj = line_next.project(p_entry)
            # 稍微取一点 epsilon
            p_ahead = line_next.interpolate(min(line_next.length, dist_proj + 0.001)) # 0.001度 approx 100m
            
            vec = np.array([p_ahead.x - p_entry.x, p_ahead.y - p_entry.y])
            norm = np.linalg.norm(vec)
            if norm > 1e-9:
                dir_vec = vec / norm
            else:
                # 极其罕见情况：入口就在终点且无法延伸
                dir_vec = np.array([0., 0.])
                
            connectivity[(curr_name, next_name)] = {
                'pos': np.array([p_entry.x, p_entry.y]), # [lon, lat]
                'dir': dir_vec                           # [dx, dy]
            }
            
        return connectivity

    def get_seq_for_lane_point(self, lane_name, lon, lat):
        """找到最近的 seq (保持你原有的逻辑)"""
        # 简化版：遍历该 lane 的 df 找最近点
        rows = self.df_lanes[(self.df_lanes['lane_name'] == lane_name) & 
                             (self.df_lanes['lane_role'] == 'center')]
        if rows.empty: return 1
        
        # 计算距离
        dists = (rows['lon'] - lon)**2 + (rows['lat'] - lat)**2
        idx = dists.idxmin()
        return rows.loc[idx, 'seq']

    def find_nearest_lane(self, lon, lat):
        """找到最近的航道 (保持你原有的逻辑)"""
        p = Point(lon, lat)
        best_name = None
        min_dist = float('inf')
        
        for name, data in self.lanes.items():
            dist = data['center'].distance(p)
            if dist < min_dist:
                min_dist = dist
                best_name = name
        return best_name

    def _get_branch_features(self, curr_pos, curr_dir_vec, curr_lane, next_lanes):
        """
        计算 "Where is next" 的 3D 特征
        Return: [MAX_BRANCHES, 3]
        """
        feats = np.zeros((self.max_branches, 3), dtype=np.float32)
        
        # 遍历候选列表
        for i, next_lane in enumerate(next_lanes):
            if i >= self.max_branches: break
            
            dist = 0.0
            rel_angle = 0.0
            rel_heading = 0.0
            
            # Case A: 保持当前航道 (Keep Lane) -> 特征全为 0
            if next_lane == curr_lane:
                pass 
                
            # Case B: 换道/分叉
            else:
                key = (curr_lane, next_lane)
                if key in self.connectivity:
                    conn = self.connectivity[key]
                    target_pos = conn['pos'] # [lon, lat]
                    target_dir = conn['dir'] # [dx, dy]
                    
                    # 1. 距离 (Dist)
                    diff = target_pos - curr_pos
                    raw_dist = np.linalg.norm(diff)
                    # 缩放一下，让数值不那么小 (0.001 deg -> 0.1)
                    dist = raw_dist * 100.0 
                    
                    # 2. 相对方位 (Angle)
                    angle_to_target = np.arctan2(diff[1], diff[0])
                    curr_angle = np.arctan2(curr_dir_vec[1], curr_dir_vec[0])
                    
                    delta_angle = angle_to_target - curr_angle
                    # 归一化到 [-pi, pi]
                    rel_angle = (delta_angle + np.pi) % (2*np.pi) - np.pi
                    
                    # 3. 走向差异 (Heading Diff)
                    target_lane_angle = np.arctan2(target_dir[1], target_dir[0])
                    delta_heading = target_lane_angle - curr_angle
                    rel_heading = (delta_heading + np.pi) % (2*np.pi) - np.pi
            
            feats[i] = [dist, rel_angle, rel_heading]
            
        return feats

    def __getitem__(self, idx):
        # 1. 获取原始数据
        seq_x_norm = self.seq_x[idx] # [T_in, 6]
        mask = self.seq_x_mask[idx]  # [T_in]
        
        # 2. 反归一化得到物理坐标 (用于几何计算)
        min_v = self.vals_to_norm['min_vals']
        max_v = self.vals_to_norm['max_vals']
        
        # x_phys: [T_in, 6] (Lon, Lat, SOG, COG, ...)
        x_phys = seq_x_norm * (max_v - min_v) + min_v
        
        # 3. 准备容器
        lane_feats_list = []      # s_norm, d_signed
        lane_dir_feats_list = []  # current_lane_dir (2D)
        next_geom_feats_list = [] # [MAX_BRANCHES * 3] -> new!
        
        T_in = self.t_in
        
        for t in range(T_in):
            # 如果是 padding 部分，直接填 0
            if mask[t] == 0:
                lane_feats_list.append([0, 0])
                lane_dir_feats_list.append([0, 0])
                next_geom_feats_list.append(np.zeros(self.max_branches * 3))
                continue
                
            lon = x_phys[t, 0]
            lat = x_phys[t, 1]
            p = Point(lon, lat)
            
            # --- A. 基础航道匹配 ---
            best_lane = self.find_nearest_lane(lon, lat)
            
            # 计算 s, d (简化的投影逻辑)
            # ... (假设你这里的 project 逻辑已经有了，这里简写) ...
            if best_lane:
                line = self.lanes[best_lane]['center']
                s_proj = line.project(p, normalized=True) # [0, 1]
                p_proj = line.interpolate(s_proj * line.length)
                
                # 计算 d (简单欧式距离，带符号需额外逻辑，此处略)
                d_val = line.distance(p) * 100.0 # scale up
                
                lane_feats_list.append([s_proj, d_val])
                
                # --- B. 计算当前航道方向 (Current Lane Dir) ---
                # 取切线
                p_next_step = line.interpolate(min(line.length, s_proj * line.length + 0.001))
                dx = p_next_step.x - p_proj.x
                dy = p_next_step.y - p_proj.y
                norm = math.sqrt(dx*dx + dy*dy)
                if norm > 0:
                    curr_dir_vec = np.array([dx/norm, dy/norm])
                else:
                    curr_dir_vec = np.array([1.0, 0.0]) # fallback
                
                lane_dir_feats_list.append(curr_dir_vec)
                
                # --- C. 【新功能】计算 Next Lane Geometry ---
                # 1. 找 seq
                seq = self.get_seq_for_lane_point(best_lane, lon, lat)
                
                # 2. 查 Next Lane 列表 (从 dataframe 中取)
                # 找到对应行
                row = self.df_lanes[(self.df_lanes['lane_name'] == best_lane) & 
                                    (self.df_lanes['seq'] == seq)]
                
                if not row.empty:
                    next_lanes_list = row.iloc[0]['next_lane'] # 已经是 list
                else:
                    next_lanes_list = []
                
                # 3. 计算几何特征 [2, 3]
                geom_feats_block = self._get_branch_features(
                    curr_pos=np.array([lon, lat]),
                    curr_dir_vec=curr_dir_vec,
                    curr_lane=best_lane,
                    next_lanes=next_lanes_list
                )
                
                # 展平 [6]
                next_geom_feats_list.append(geom_feats_block.flatten())
                
            else:
                # 没找到航道 (异常处理)
                lane_feats_list.append([0, 0])
                lane_dir_feats_list.append([0, 0])
                next_geom_feats_list.append(np.zeros(self.max_branches * 3))

        # 4. 拼接
        lane_feats = np.array(lane_feats_list)       # [T, 2]
        lane_dir_feats = np.array(lane_dir_feats_list) # [T, 2]
        next_geom_feats = np.array(next_geom_feats_list) # [T, 6]
        
        # 最终特征组合: 
        # [seq_x (6), lane_feats (2), lane_dir (2), next_geom (6)] => Total 16 dim
        seq_x_final = np.concatenate([seq_x_norm, lane_feats, lane_dir_feats, next_geom_feats], axis=-1)
        
        # seq_y, masks 不需要动
        return {
            'seq_x': torch.FloatTensor(seq_x_final),
            'seq_y': torch.FloatTensor(self.seq_y[idx]),
            'seq_x_mask': torch.FloatTensor(mask),
            'seq_y_mask': torch.FloatTensor(self.seq_y_mask[idx])
        }

    def __len__(self):
        return len(self.seq_x)