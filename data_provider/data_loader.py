import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import warnings

warnings.filterwarnings('ignore')


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
        predict_position_only=True
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
        self.social_sigma = 1.0  # 社交图中的速度差异标准差参数
        
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
            
            print(f'\nNormalization parameters ({self.scale_type}):')
            feature_names = ['Longitude', 'Latitude', 'COG', 'SOG']
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
        A_social_enc = self._build_social_graph(seq_x)  # [T_in, N, N]
        # A_social_dec = self._build_social_graph(seq_y)

        
        # 如果只预测经纬度，可以在这里标记
        # 但为了保持数据格式一致，我们仍然返回完整的4个特征
        # 在计算损失时再决定使用哪些特征
        
        return seq_x, seq_y, seq_x_mask, seq_y_mask, ship_count, global_id, A_social_enc
    
    def __len__(self):
        return len(self.data_x)

    def _build_social_graph(self, x_data):
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

        # d. 处理对角线(i=j)
        diag_idx = np.arange(N)
        A_social[:, diag_idx, diag_idx] = 0.0

        return A_social # [T_seq, N, N]







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