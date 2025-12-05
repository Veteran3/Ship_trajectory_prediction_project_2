import os
import time
import warnings
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, visual
from utils.metrics import metric, ADE, FDE

# 引入 ReduceLROnPlateau
from torch.optim.lr_scheduler import ReduceLROnPlateau

warnings.filterwarnings('ignore')


class Exp_Forecasting(Exp_Basic):
    def __init__(self, args):
        super(Exp_Forecasting, self).__init__(args)
    
    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model
    
    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def vali(self, vali_loader):
        total_loss = []
        total_loss_absolute = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, mask_x, mask_y, ship_count, _, A_social, edge_features) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                mask_x = mask_x.to(self.device)
                mask_y = mask_y.to(self.device)
                batch_y = batch_y.float().to(self.device)

                outputs = self.model(
                    x_enc=batch_x, 
                    y_truth_abs=None,
                    mask_x=mask_x, 
                    mask_y=mask_y,
                    A_social_t=A_social.to(self.device),
                    edge_features=edge_features.to(self.device),
                ) 
                
                loss, loss_absolute,_ = self.model.loss(
                    pred=outputs,
                    y_truth_abs=batch_y,
                    x_enc=batch_x,
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
        训练模型 - 支持接续训练和保存优化器状态
        """
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')
        
        ckpt_path = os.path.join(setting, 'checkpoints')
        if not os.path.exists(ckpt_path):
            os.makedirs(ckpt_path, exist_ok=True)
        
        time_now = time.time()
        
        train_steps = len(train_loader)
        
        # 1. 初始化优化器
        model_optim = self._select_optimizer()
        
        # 2. [修改] 初始化 LR Scheduler (ReduceLROnPlateau)
        scheduler = ReduceLROnPlateau(
            model_optim,
            mode='min',
            factor=0.5,      # 每次减半
            patience=3,      # 连续 3 轮 vali loss 不下降才减
            min_lr=1e-6,
            # verbose=True
        )

        # 3. [新增] 接续训练逻辑 (Resume)
        start_epoch = 0
        best_vali_loss = float('inf')
        
        # 检查是否需要 Resume (依据 args.exp_setting 是否有值，或者 args.resume 标记)
        # 这里假设如果传入的 setting 对应的目录下已经有 checkpoint，且 args.resume 为 True 或 exp_setting 非空，则尝试加载
        resume_checkpoint_path = os.path.join(ckpt_path, 'checkpoint.pth')
        
        if (hasattr(self.args, 'exp_setting') and self.args.exp_setting) or (hasattr(self.args, 'resume') and self.args.resume):
            if os.path.exists(resume_checkpoint_path):
                print(f"Resuming training from {resume_checkpoint_path}")
                checkpoint = torch.load(resume_checkpoint_path, map_location=self.device)
                
                # 加载模型权重
                if 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    self.model.load_state_dict(checkpoint) # 兼容旧版

                # 加载优化器和Scheduler (如果存在)
                if 'optimizer_state_dict' in checkpoint:
                    model_optim.load_state_dict(checkpoint['optimizer_state_dict'])
                if 'scheduler_state_dict' in checkpoint:
                    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                
                # 恢复 Epoch
                if 'epoch' in checkpoint:
                    start_epoch = checkpoint['epoch'] + 1
                    print(f"Resuming from epoch {start_epoch}")
                
                # 恢复最佳 Loss (用于 EarlyStopping 判断)
                if 'best_loss' in checkpoint:
                    best_vali_loss = checkpoint['best_loss']

        # 初始化 EarlyStopping
        # 注意：为了保存完整的优化器状态，我们不再完全依赖 EarlyStopping 内部的 save，
        # 而是利用它的 counter 来判断是否需要保存 best model
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        # 如果是接续训练，需要手动更新 early_stopping 的内部状态
        if best_vali_loss != float('inf'):
            early_stopping.best_score = -best_vali_loss # EarlyStopping 内部使用负值作为 score
        
        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()
        
        for epoch in range(start_epoch, self.args.train_epochs):
            iter_count = 0
            train_loss = []
            train_loss_absolute = []
            
            self.model.train()
            epoch_time = time.time()
            
            for i, (batch_x, batch_y, mask_x, mask_y, ship_count, _, A_social, edge_features) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                mask_x = mask_x.to(self.device)
                mask_y = mask_y.to(self.device)
                batch_y = batch_y.float().to(self.device)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        pred_deltas = self.model(
                            x_enc=batch_x,
                            y_truth_abs=batch_y,
                            mask_x=mask_x,
                            mask_y=mask_y,
                            A_social_t=A_social.to(self.device)
                        )
                        loss, loss_absolute = self.model.loss(
                            pred_deltas=pred_deltas,
                            y_truth_abs=batch_y,
                            x_enc=batch_x,
                            mask_y=mask_y
                        )
                        
                        back_loss = 0.5 * loss + loss_absolute 
                        
                        train_loss.append(loss.item())
                        train_loss_absolute.append(loss_absolute.item())
                else:
                    pred = self.model(
                        x_enc=batch_x,
                        y_truth_abs=batch_y[..., :2],
                        mask_x=mask_x,
                        mask_y=mask_y,
                        A_social_t=A_social.to(self.device),
                        edge_features=edge_features.to(self.device),
                    )
                    
                    loss, loss_absolute, loss_start = self.model.loss(
                        pred=pred,
                        y_truth_abs=batch_y,
                        x_enc=batch_x,
                        mask_y=mask_y
                    )
                    train_loss.append(loss.item())
                    train_loss_absolute.append(loss_absolute.item())
                
                if (i + 1) % 100 == 0:
                    print(f"\titers: {i + 1}, epoch: {epoch + 1} | loss delta: {loss.item():.7f},  loss absolute: {loss_absolute.item():.7f}")
                
                if self.args.use_amp:
                    scaler.scale(back_loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    back_loss = loss + 0.5 * loss_absolute + 2.0 * loss_start
                    back_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    model_optim.step()
            
            print(f"Epoch: {epoch + 1} cost time: {time.time() - epoch_time}")
            train_loss = np.average(train_loss)
            train_loss_absolute = np.average(train_loss_absolute)
            
            vali_loss, vali_loss_absolute = self.vali(vali_loader)
            test_loss, test_loss_absolute = self.vali(test_loader)

            print(f"Epoch: {epoch + 1}, Steps: {train_steps} | Train Loss: {train_loss:.7f}, {train_loss_absolute:.7f}, Vali Loss: {vali_loss:.7f}, {vali_loss_absolute:.7f} Test Loss: {test_loss:.7f}, {test_loss_absolute:.7f}")
            
            # [修改] 学习率调整策略 Update Scheduler
            # 注意：ReduceLROnPlateau 需要传入当前的 validation metric
            scheduler.step(vali_loss_absolute)

            # [新增] 构造包含所有信息的 Checkpoint 字典
            checkpoint_dict = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': model_optim.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_loss': early_stopping.best_score if early_stopping.best_score != -np.Inf else -vali_loss_absolute,
                'args': self.args
            }

            # [修改] Early Stopping 与 模型保存逻辑
            # 调用 early_stopping 计算 counter
            early_stopping(vali_loss_absolute, self.model, ckpt_path)
            
            # 如果当前是最佳模型 (counter 为 0)，覆盖保存完整的 checkpoint
            # 注意：EarlyStopping 内部可能已经保存了一个只包含 state_dict 的文件，
            # 我们这里将其覆盖为包含优化器的完整 checkpoint
            if early_stopping.counter == 0:
                print(f"Saving full checkpoint with optimizer to {os.path.join(ckpt_path, 'checkpoint.pth')}")
                torch.save(checkpoint_dict, os.path.join(ckpt_path, 'checkpoint.pth'))
            
            # 可选：每个 epoch 结束都保存一个 latest checkpoint，方便随时断点续传（即使不是最优模型）
            torch.save(checkpoint_dict, os.path.join(ckpt_path, 'checkpoint_latest.pth'))

            if early_stopping.early_stop:
                print("Early stopping")
                break
            
        # 加载最佳模型进行返回
        best_model_path = os.path.join(ckpt_path, 'checkpoint.pth')
        # [修改] 加载时处理字典结构
        checkpoint = torch.load(best_model_path)
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        
        results_path = os.path.join(setting, 'results')
        if not os.path.exists(results_path):
            os.makedirs(results_path, exist_ok=True)
            
        if test:
            print('loading model')
            checkpoint_path = os.path.join(setting, 'checkpoints', 'checkpoint.pth')
            # [修改] 兼容加载完整 Checkpoint 字典
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)

        hists = []
        preds = []
        trues = []
        masks_list = []
        
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, mask_x, mask_y, ship_count, _, A_social, edge_features) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                outputs = self.model(
                    x_enc=batch_x, 
                    y_truth_abs=None,
                    mask_x=mask_x.to(self.device), 
                    mask_y=mask_y.to(self.device),
                    A_social_t=A_social.to(self.device),
                    edge_features=edge_features.to(self.device),
                )
                
                hists.append(batch_x[..., :2].detach().cpu().numpy())
                preds.append(outputs.detach().cpu().numpy())
                trues.append(batch_y[..., :2].detach().cpu().numpy())
                masks_list.append(mask_y.detach().cpu().numpy())
        
        scaler_file = os.path.join(self.args.root_path, 'scaler_params.npy')
        print('Loading scaler parameters from:', scaler_file)
        scaler_params = np.load(scaler_file, allow_pickle=True).item()
        mean, std = scaler_params['mean'], scaler_params['std']

        hists = np.concatenate(hists, axis=0)
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        masks = np.concatenate(masks_list, axis=0).astype(bool)

        print('hists shape:', hists.shape)
        print('preds shape:', preds.shape)
        print('trues shape:', trues.shape)

        hists_invers = hists * std[:2] + mean[:2]
        preds_invers = preds * std[:2] + mean[:2]
        trues_invers = trues * std[:2] + mean[:2]
        
        print('test shape:', preds.shape, trues.shape, masks.shape)
        
        ade = ADE(preds, trues, mask=masks)
        fde = FDE(preds, trues, mask=masks)
        mae, mse, rmse, mape, mspe = metric(preds, trues, mask=masks)
        
        print(f'Metrics (on valid data only):')
        print(f'mse:{mse:.4f}, mae:{mae:.4f}, rmse:{rmse:.4f}, mape:{mape:.4f}, mspe:{mspe:.4f}')
        print(f'ADE:{ade:.4f}, FDE:{fde:.4f}')
        
        with open(os.path.join(results_path, 'result.txt'), 'w') as f:
            f.write(setting + '\n')
            f.write(f'ADE: {ade:.4f}, FDE: {fde:.4f}\n')
            f.write(f'mse:{mse:.4f}, mae:{mae:.4f}, rmse:{rmse:.4f}, mape:{mape:.4f}, mspe:{mspe:.4f}\n')
        
        np.save(os.path.join(results_path, 'metrics.npy'), np.array([mae, mse, rmse, mape, mspe, ade, fde]))
        np.save(os.path.join(results_path, 'hists.npy'), hists)
        np.save(os.path.join(results_path, 'pred.npy'), preds)
        np.save(os.path.join(results_path, 'true.npy'), trues)
        np.save(os.path.join(results_path, 'mask.npy'), masks)
        np.save(os.path.join(results_path, 'hists_inverse.npy'), hists_invers)
        np.save(os.path.join(results_path, 'pred_inverse.npy'), preds_invers)
        np.save(os.path.join(results_path, 'true_inverse.npy'), trues_invers)
        
        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='test')
        
        results_path = os.path.join('./results/', setting)
        if not os.path.exists(results_path):
            os.makedirs(results_path, exist_ok=True)
            
        if load:
            checkpoint_path = os.path.join(self.args.checkpoints, setting, 'checkpoint.pth')
            # [修改] 兼容加载完整 Checkpoint 字典
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
        
        preds = []
        
        self.model.eval()
        with torch.no_grad():
            for i,(batch_x, batch_y, mask_x, mask_y, ship_count, _) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                
                outputs = self.model(
                    x_enc=batch_x, 
                    y_truth_abs=None,
                    mask_x=mask_x.to(self.device), 
                    mask_y=mask_y.to(self.device)
                )
                
                preds.append(outputs.detach().cpu().numpy())
        
        preds = np.concatenate(preds, axis=0)
        np.save(os.path.join(results_path, 'real_prediction.npy'), preds)
        
        return preds