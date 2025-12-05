import os
import time
import warnings
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric, ADE, FDE

# 1. 引入 ReduceLROnPlateau
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
                
                loss, loss_absolute = self.model.loss(
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
        训练模型 - 修复 torch.load 报错，并增加 LR 打印
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
        
        # 2. 初始化 LR Scheduler
        scheduler = ReduceLROnPlateau(
            model_optim,
            mode='min',
            factor=0.5,
            patience=0,
            min_lr=1e-7,
        )

        # 初始化 EarlyStopping
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        # 3. 接续训练逻辑
        start_epoch = 0
        best_epoch_loss = float('inf')
        # 定义文件路径
        best_model_path = os.path.join(ckpt_path, 'checkpoint.pth')
        latest_model_path = os.path.join(ckpt_path, 'checkpoint_latest.pth')

        if hasattr(self.args, 'resume') and self.args.resume == 1:
            # [修正] 优先加载 latest，如果不存在再尝试加载 best
            resume_path = None
            if os.path.exists(latest_model_path):
                resume_path = latest_model_path
                print(f"RESUME SIGNAL DETECTED: Loading LATEST checkpoint from {resume_path}")
            elif os.path.exists(best_model_path):
                resume_path = best_model_path
                print(f"RESUME SIGNAL DETECTED: Latest not found, loading BEST checkpoint from {resume_path}")
            
            if resume_path:
                # [Fix] weights_only=False
                checkpoint = torch.load(resume_path, map_location=self.device, weights_only=False)
                
                # (1) 恢复模型权重
                if 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    self.model.load_state_dict(checkpoint) 

                # (2) 恢复优化器状态
                if 'optimizer_state_dict' in checkpoint:
                    model_optim.load_state_dict(checkpoint['optimizer_state_dict'])
                    print("Optimizer state loaded.")

                # (3) 恢复 Scheduler 状态
                if 'scheduler_state_dict' in checkpoint:
                    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                    print("Scheduler state loaded.")
                    scheduler.patience = 0  # 重置 patience，避免过早调整学习率
                    print(f"Scheduler patience reset to {scheduler.patience}")
                
                # (4) 恢复 Epoch
                if 'epoch' in checkpoint:
                    start_epoch = checkpoint['epoch'] + 1
                    print(f"Resuming training from epoch {start_epoch + 1}")
                
                # (5) 恢复 EarlyStopping 的最佳记录
                # 如果我们加载的是 latest，它的 loss 可能不是 best，所以我们需要读取记录的 best_loss
                if 'best_loss' in checkpoint:
                    best_epoch_loss = checkpoint['best_loss']
                    early_stopping.best_score = -best_epoch_loss
                    print(f"Restored best loss for EarlyStopping: {best_epoch_loss:.7f}")
            else:
                print(f"Resume signal is 1, but no checkpoint files found. Starting from scratch.")

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()
        
        for epoch in range(start_epoch, self.args.train_epochs):
            # [新增] 在每一轮开始时打印当前学习率
            current_lr = model_optim.param_groups[0]['lr']
            
            print(f">>>>> Epoch: {epoch + 1}, Current Learning Rate: {current_lr:.8f} <<<<<")

            iter_count = 0
            train_loss = []
            train_loss_absolute = []
            
            self.model.train()
            epoch_time = time.time()
            
            if epoch < 4:
                self.model.sampling_prob = 1.0
            elif epoch >= 4 and epoch < 8:
                self.model.sampling_prob = 0.7
            else:
                self.model.sampling_prob = 0.5

            print(f"[INFO] Sampling Probability: {self.model.sampling_prob:.3f}")

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
                    
                    back_loss, loss_absolute = self.model.loss(
                        pred=pred,
                        y_truth_abs=batch_y,
                        x_enc=batch_x,
                        mask_y=mask_y,
                        iter=i,
                        epoch=epoch
                    )
                    train_loss.append(back_loss.item())
                    train_loss_absolute.append(loss_absolute.item())
                
                if self.args.use_amp:
                    scaler.scale(back_loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    back_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    model_optim.step()
            
            print(f"Epoch: {epoch + 1} cost time: {time.time() - epoch_time}")
            train_loss = np.average(train_loss)
            train_loss_absolute = np.average(train_loss_absolute)

            vali_loss, vali_loss_absolute = self.vali(vali_loader)
            test_loss, test_loss_absolute = self.vali(test_loader)

            print(f"Epoch: {epoch + 1}, Steps: {train_steps} | Train Loss: {train_loss:.7f}, {train_loss_absolute:.7f}, Vali Loss: {vali_loss:.7f}, {vali_loss_absolute:.7f} Test Loss: {test_loss:.7f}, {test_loss_absolute:.7f}")
            # 更新 Scheduler
            scheduler.step(vali_loss_absolute)

            # 更新历史最佳 Loss (用于保存到 checkpoint 中)
            if vali_loss_absolute < best_epoch_loss:
                best_epoch_loss = vali_loss_absolute

            # ------------------------------------------------------------
            # [修正] 保存 Latest 模型 (每轮结束都保存，用于接续训练)
            # ------------------------------------------------------------
            latest_checkpoint_dict = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': model_optim.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_loss': best_epoch_loss, # 始终记录历史最佳，以便 EarlyStopping 恢复
                'args': self.args
            }
            torch.save(latest_checkpoint_dict, latest_model_path)
            print(f"Saving LATEST checkpoint to {latest_model_path}")
            # ------------------------------------------------------------

            # 调用 EarlyStopping
            early_stopping(vali_loss_absolute, self.model, ckpt_path)

            # 如果当前是最佳模型，覆盖 checkpoint.pth
            if early_stopping.counter == 0:
                print(f"Saving BEST checkpoint to {best_model_path}")
                # 我们复用 latest 字典，因为它们的内容在当前这一刻是一样的
                torch.save(latest_checkpoint_dict, best_model_path)

            if early_stopping.early_stop:
                print("Early stopping")
                break
            
        # 训练结束，加载最佳模型进行返回 (用于后续测试)
        print(f"Loading best model for testing from {best_model_path}")
        # [Fix] weights_only=False
        checkpoint = torch.load(best_model_path, weights_only=False)
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
            # [Fix] 添加 weights_only=False
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
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
        # [Fix] np.load 也可能受影响，不过通常 allow_pickle=True 就够了，这里无需修改 weights_only 因为是 numpy 加载
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
            # [Fix] 添加 weights_only=False
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
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