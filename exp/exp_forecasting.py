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
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric

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

    def vali(self, vali_data, vali_loader, criterion):
        """
        验证模型
        
        Args:
            vali_data: 验证数据集
            vali_loader: 验证数据加载器
            criterion: 损失函数
        
        Returns:
            total_loss: 总损失
        """
        total_loss = []
        self.model.eval()
        
        with torch.no_grad():
            for i, (batch_x, batch_y, mask_x, mask_y, ship_count, _) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)  # [B, T_in, N, D]
                batch_y = batch_y[..., :2].float().to(self.device)  # [B, T_out, N, 2]
                
                # 模型推理（不使用teacher forcing）
                outputs = self.model(batch_x, x_dec=None, mask_x=mask_x.to(self.device), mask_y=mask_y.to(self.device))  # [B, T_out, N, 2]
                
                # 计算损失
                loss = criterion(outputs[mask_y], batch_y[mask_y])
                total_loss.append(loss.item())
        
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss
    
    def train(self, setting):
        """
        训练模型
        
        Args:
            setting: 实验设置名称
        """
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')
        
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)
        
        time_now = time.time()
        
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        
        model_optim = self._select_optimizer()
        

        criterion = self.model.get_loss(self.args.loss)
        
        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()
        
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            
            self.model.train()
            epoch_time = time.time()

            for i, (batch_x, batch_y, mask_x, mask_y, ship_count, _) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                # print("data x:", batch_x[0, 0, 0, :2])

                batch_x = batch_x.float().to(self.device)  # [B, T_in, N, D]
                batch_y = batch_y[..., :2].float().to(self.device)  # [B, T_out, N, 2]
                
                # print("batch_x:", batch_x[0, 0, -1, :])
                # print("batch_y:", batch_y[0, 0, -1, :])

                # 前向传播（使用teacher forcing）
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, x_dec=batch_y)

                        # 打印 outputs 每个维度的 max / min / mean
                        t = outputs.detach().cpu()
                        # 整体统计
                        print(f"outputs overall -> max: {t.max().item():.6f}, min: {t.min().item():.6f}, mean: {t.mean().item():.6f}")
                        # 按最后一维（feature 维）统计：将前面维度合并后对每个 feature 计算
                        feat_dim = t.size(-1)
                        flat = t.view(-1, feat_dim)
                        max_per_feat = flat.max(dim=0)[0].numpy()
                        min_per_feat = flat.min(dim=0)[0].numpy()
                        mean_per_feat = flat.mean(dim=0).numpy()
                        print(f"outputs per-dim -> max: {max_per_feat}, min: {min_per_feat}, mean: {mean_per_feat}")

                        # print("outputs shape:", outputs.shape)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    outputs = self.model(batch_x, x_dec=batch_y, mask_x=mask_x.to(self.device), mask_y=mask_y.to(self.device))

                    # # 打印 outputs 每个维度的 max / min / mean
                    # t = outputs.detach().cpu()
                    # # 整体统计
                    # print(f"outputs overall -> max: {t.max().item():.6f}, min: {t.min().item():.6f}, mean: {t.mean().item():.6f}")
                    # # 按最后一维（feature 维）统计：将前面维度合并后对每个 feature 计算
                    # feat_dim = t.size(-1)
                    # flat = t.view(-1, feat_dim)
                    # max_per_feat = flat.max(dim=0)[0].numpy()
                    # min_per_feat = flat.min(dim=0)[0].numpy()
                    # mean_per_feat = flat.mean(dim=0).numpy()
                    # print(f"outputs per-dim -> max: {max_per_feat}, min: {min_per_feat}, mean: {mean_per_feat}")

                    # print("outputs shape:", outputs.shape)

                    # print("outputs shape:", outputs.shape)
                    loss = criterion(outputs[mask_y], batch_y[mask_y])
                    train_loss.append(loss.item())
                
                # 打印训练信息
                if (i + 1) % 100 == 0:
                    print(f"\titers: {i + 1}, epoch: {epoch + 1} | loss: {loss.item():.7f}")
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print(f'\tspeed: {speed:.4f}s/iter; left time: {left_time:.4f}s')
                    iter_count = 0
                    time_now = time.time()
                
                # 反向传播
                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    model_optim.step()
            
            print(f"Epoch: {epoch + 1} cost time: {time.time() - epoch_time}")
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)
            
            print(f"Epoch: {epoch + 1}, Steps: {train_steps} | Train Loss: {train_loss:.7f} Vali Loss: {vali_loss:.7f} Test Loss: {test_loss:.7f}")
            
            # Early stopping
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            
            # 调整学习率
            adjust_learning_rate(model_optim, epoch + 1, self.args)
        
        # 加载最佳模型
        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        
        return self.model
    
    def test(self, setting, test=0):
        """
        测试模型
        
        Args:
            setting: 实验设置名称
            test: 0-验证集, 1-测试集
        """
        test_data, test_loader = self._get_data(flag='test')
        
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))
        
        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, mask_x, mask_y, ship_count, _) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)  # [B, T_in, N, D]
                batch_y = batch_y[..., :2].float().to(self.device)  # [B, T_out, N, 2]
                
                # 模型推理
                outputs = self.model(batch_x, x_dec=None, mask_x=mask_x.to(self.device), mask_y=mask_y.to(self.device))  # [B, T_out, N, 2]
                
                # 收集预测和真实值
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                
                preds.append(outputs)
                trues.append(batch_y)
        
        # 拼接所有批次
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        print('test shape:', preds.shape, trues.shape)
        
        # 保存结果
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        # 计算指标
        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print(f'mse:{mse:.4f}, mae:{mae:.4f}, rmse:{rmse:.4f}, mape:{mape:.4f}, mspe:{mspe:.4f}')
        
        # 保存结果到文件
        with open(os.path.join(folder_path, 'result.txt'), 'w') as f:
            f.write(setting + '\n')
            f.write(f'mse:{mse:.4f}, mae:{mae:.4f}, rmse:{rmse:.4f}, mape:{mape:.4f}, mspe:{mspe:.4f}\n')
        
        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)
        
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