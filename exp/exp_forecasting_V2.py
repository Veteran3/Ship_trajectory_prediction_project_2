import os
import time
import warnings
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

from data_provider.data_loader import ShipTrajectoryDataset

from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual, get_annealed_sampling_prob
from utils.metrics import metric, ADE, FDE

warnings.filterwarnings('ignore')


class Exp_Forecasting(Exp_Basic):
    def __init__(self, args):
        super(Exp_Forecasting, self).__init__(args)
    
    def _build_model(self):
        """
        构建模型
        """
        model = self.model_dict[self.args.model].Model(self.args).float()

        
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        
        return model
    
    def _get_data(self, flag):
        """
        获取数据加载器
        
        Args:
            flag: 'train', 'val', 'test'
        
        Returns:
            data_set, data_loader
        """
        args = self.args
        
        # 数据集参数
        data_dict = {
            'train': (args.train_data_path, True),
            'val': (args.val_data_path, True),
            'test': (args.test_data_path, False)
        }
        
        data_path, shuffle_flag = data_dict[flag]
        
        # 根据你的数据格式选择合适的Dataset
        # 这里假设你有一个ShipTrajectoryDataset
        
        
        data_set = ShipTrajectoryDataset(
            root_path=args.root_path,
            data_path=data_path,
            flag=flag,
            size=[args.seq_len, args.pred_len],
            num_ships=args.num_ships,
            num_features=args.num_features,
            scale=args.scale
        )
        
        print(f'{flag} dataset size: {len(data_set)}')
        
        data_loader = DataLoader(
            data_set,
            batch_size=args.batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=True
        )
        
        return data_set, data_loader
    
    def _select_optimizer(self):
        """选择优化器"""
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim
    def vali(self, vali_loader):
        """
        验证模型 (已修正)
        """
        total_loss = []
        total_loss_absolute = []
        self.model.eval()
        
        with torch.no_grad():
            # 您的原始 loader 签名
            for i, (batch_x, batch_y, mask_x, mask_y, ship_count, _) in enumerate(vali_loader):
                
                batch_x = batch_x.float().to(self.device)  # [B, T_in, N, D_in] (x_enc)
                mask_x = mask_x.to(self.device)
                mask_y = mask_y.to(self.device)
                # batch_y 是 y_truth_abs
                batch_y = batch_y[..., :2].float().to(self.device)  # [B, T_out, N, 2] (y_truth_abs)
                
                # 1. 模型推理 (y_truth_abs=None 触发推理)
                outputs_deltas = self.model(
                    x_enc=batch_x, 
                    y_truth_abs=None, # <-- 触发推理
                    mask_x=mask_x, 
                    mask_y=mask_y
                ) 
                
                # 2. 计算损失
                loss, loss_absolute = self.model.loss(
                    pred_deltas=outputs_deltas,
                    y_truth_abs=batch_y, # 传入真实绝对值
                    x_enc=batch_x,       # 传入历史
                    mask_y=mask_y
                )
                
                total_loss.append(loss.item())
                total_loss_absolute.append(loss_absolute.item())
        
        total_loss = np.average(total_loss)
        total_loss_absolute = np.average(total_loss_absolute)
        self.model.train()
        return total_loss, total_loss_absolute
    
    def train(self, setting):
        """
        训练模型 (已修正)
        """
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')
        
        # ... (path, time_now, early_stopping, model_optim 不变) ...
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)
        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        model_optim = self._select_optimizer()

        # [移除 criterion]
        
        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()
        
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            train_loss_absolute = []
            
            self.model.train()
            epoch_time = time.time()

            # (预定采样退火 - 不变)
            current_prob = get_annealed_sampling_prob(self.args, epoch)
            self.model.sampling_prob = current_prob
            # (我移除了打印，保持和您代码一致)

            # 您的原始 loader 签名
            for i, (batch_x, batch_y, mask_x, mask_y, ship_count, _) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)  # [B, T_in, N, D_in] (x_enc)
                mask_x = mask_x.to(self.device)
                mask_y = mask_y.to(self.device)
                # batch_y 是 y_truth_abs
                batch_y = batch_y[..., :2].float().to(self.device)  # [B, T_out, N, 2] (y_truth_abs)
                
                # 前向传播
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        # [关键修正]
                        pred_deltas = self.model(
                            x_enc=batch_x,
                            y_truth_abs=batch_y, # <-- 传入 "真实绝对"
                            mask_x=mask_x,
                            mask_y=mask_y
                        )
                        
                        loss, loss_absolute = self.model.loss(
                            pred_deltas=pred_deltas,
                            y_truth_abs=batch_y,
                            x_enc=batch_x,
                            mask_y=mask_y
                        )
                        train_loss.append(loss.item())
                else:
                    # [关键修正]
                    pred_deltas = self.model(
                        x_enc=batch_x,
                        y_truth_abs=batch_y,
                        mask_x=mask_x,
                        mask_y=mask_y
                    )
                    
                    loss, loss_absolute = self.model.loss(
                        pred_deltas=pred_deltas,
                        y_truth_abs=batch_y,
                        x_enc=batch_x,
                        mask_y=mask_y
                    )
                    train_loss.append(loss.item())
                    train_loss_absolute.append(loss_absolute.item())
                
                # (打印训练信息 - 不变)
                if (i + 1) % 100 == 0:
                    print(f"\titers: {i + 1}, epoch: {epoch + 1} | loss delta: {loss.item():.7f},  loss absolute: {loss_absolute.item():.7f}")
                    # ... (speed, left_time) ...
                
                # (反向传播 - 不变)
                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    back_loss = loss + 0.5 * loss_absolute
                    back_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    model_optim.step()
            
            print(f"Epoch: {epoch + 1} cost time: {time.time() - epoch_time}")
            train_loss = np.average(train_loss)
            train_loss_absolute = np.average(train_loss_absolute)
            
            # [关键修正]
            vali_loss, vali_loss_absolute = self.vali(vali_loader)
            test_loss, test_loss_absolute = self.vali(test_loader) # (假设 test 也用 vali 逻辑)
            
            print(f"Epoch: {epoch + 1}, Steps: {train_steps} | Train Loss: {train_loss:.7f}, {train_loss_absolute:.7f} Vali Loss: {vali_loss:.7f}, {vali_loss_absolute:.7f} Test Loss: {test_loss:.7f}, {test_loss_absolute:.7f}")
            
            # (Early stopping, adjust_learning_rate - 不变)
            early_stopping(vali_loss_absolute, self.model, path)
            if early_stopping.early_stop:
                break
            adjust_learning_rate(model_optim, epoch + 1, self.args)
        
        # (加载最佳模型 - 不变)
        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        
        return self.model

    def test(self, setting, test=0):
        """
        测试模型 (已更新为调用 model.integrate)
        """
        # 假设 loader 返回: x_enc, x_static, mask_x, mask_y, y_truth_abs
        test_data, test_loader = self._get_data(flag='test')
        
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))
        
        preds = []
        trues = []
        masks_list = [] # <-- [新] 强烈建议保存掩码
        
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        self.model.eval()
        with torch.no_grad():
            # 假设 loader 返回: x_enc, x_static, mask_x, mask_y, y_truth_abs
            for i, (batch_x, batch_y, mask_x, mask_y, ship_count, _) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)  # [B, T_in, N, D]
                batch_y = batch_y[..., :2].float().to(self.device)  # [B, T_out, N, 2]
                
                # 1. 模型推理: 获取 "预测的增量"
                outputs_deltas =  self.model(x_enc=batch_x,
                        y_truth_abs=batch_y,
                        mask_x=mask_x.to(self.device),
                        mask_y=mask_y.to(self.device))
                
                # 2. [关键修改]
                # 调用新方法，将 "增量" 重建为 "绝对坐标"
                outputs_absolute = self.model.integrate(outputs_deltas, batch_x)
                
                # 3. 收集预测和真实值
                outputs = outputs_absolute.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                
                preds.append(outputs)
                trues.append(batch_y)
                masks_list.append(mask_y.detach().cpu().numpy()) # <-- [新]
        
        # 拼接所有批次
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        masks = np.concatenate(masks_list, axis=0).astype(bool) # <-- [新]
        
        scaler_file = r'data/scaler_params.npy'
        scaler_params = np.load(scaler_file, allow_pickle=True).item()
        mean, std = scaler_params['mean'], scaler_params['std']

        # 反标准化
        preds_invers = preds * std[:2] + mean[:2]
        trues_invers = trues * std[:2] + mean[:2]



        print('test shape:', preds.shape, trues.shape, masks.shape)
        
        # 保存结果
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        # [关键修改]
        # 在 "有效" 点上计算指标
        # (这可以防止 "幽灵船" 的 0 误差拉低或 0/0 错误拉高 MAE/MAPE)
        # preds_valid = preds[masks]
        # trues_valid = trues[masks]
        ade = ADE(preds, trues, mask=masks)
        fde = FDE(preds, trues, mask=masks)
        # 重新塑形, 假设 metric 函数期望 (N_points, 2)
        # preds_valid = preds_valid.reshape(-1, 2)
        # trues_valid = trues_valid.reshape(-1, 2)

        mae, mse, rmse, mape, mspe = metric(preds, trues, mask=masks)
        
        print(f'Metrics (on valid data only):')
        print(f'mse:{mse:.4f}, mae:{mae:.4f}, rmse:{rmse:.4f}, mape:{mape:.4f}, mspe:{mspe:.4f}')
        print(f'ADE:{ade:.4f}, FDE:{fde:.4f}')
        
        # 保存结果到文件
        with open(os.path.join(folder_path, 'result.txt'), 'w') as f:
            f.write(setting + '\n')
            f.write(f'mse:{mse:.4f}, mae:{mae:.4f}, rmse:{rmse:.4f}, mape:{mape:.4f}, mspe:{mspe:.4f}\n')
        
        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe, ade, fde]))
        np.save(folder_path + 'pred.npy', preds) # 保存完整 "绝对" 预测
        np.save(folder_path + 'true.npy', trues) # 保存完整 "绝对" 真实
        np.save(folder_path + 'pred_invers.npy', preds_invers) # 保存反标准化预测
        np.save(folder_path + 'true_invers.npy', trues_invers) # 保存
        np.save(folder_path + 'mask.npy', masks) # <-- [新] 保存掩码
        
        return
    
    def predict(self, setting, load=False):
        """
        预测（用于实际部署）
        
        Args:
            setting: 实验设置名称
            load: 是否加载已保存的模型
        """
        pred_data, pred_loader = self._get_data(flag='test')
        
        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))
        
        preds = []
        
        self.model.eval()
        with torch.no_grad():
            for i,(batch_x, batch_y, mask_x, mask_y, ship_count, _) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                
                # 模型推理
                outputs = self.model(batch_x, x_dec=None)
                
                preds.append(outputs.detach().cpu().numpy())
        
        preds = np.concatenate(preds, axis=0)
        
        # 保存预测结果
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        np.save(folder_path + 'real_prediction.npy', preds)
        
        return preds