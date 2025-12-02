import os
import time
import warnings
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

"""
dataloader 版本选取指南：
v1 : 最基础版本，仅包含轨迹数据
v2 : 添加social attention
v3 : 添加航道特征
v4 : 添加下一帧航道位置特征
v5 : 修复航道特征

"""

"""
此v4版本对应v3.2.1模型。loss为四个
"""



from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual, get_annealed_sampling_prob
from utils.metrics import metric, ADE, FDE

warnings.filterwarnings('ignore')


class Exp_Forecasting(Exp_Basic):
    def __init__(self, args):
        super(Exp_Forecasting, self).__init__(args)
    
    def _build_model(self):
        # ... (此方法保持不变) ...
        model = self.model_dict[self.args.model].Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model
    
    
    
    def _select_optimizer(self):
        # ... (此方法保持不变) ...
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def vali(self, vali_loader):
        # ... (此方法保持不变, 它不处理路径) ...
        
        total_loss_list = []
        total_loss_absolute = []
        total_loss_intent = []
        total_loss_delta = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, mask_x, mask_y, ship_count, _, A_social, edge_features) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                mask_x = mask_x.to(self.device)
                mask_y = mask_y.to(self.device)
                batch_y = batch_y.float().to(self.device)

                outputs_deltas, intent_vectors = self.model(
                    x_enc=batch_x, 
                    y_truth_abs=None,
                    mask_x=mask_x, 
                    mask_y=mask_y,
                    A_social_t=A_social.to(self.device),
                    edge_features=edge_features.to(self.device),
                ) 
                
                total_loss, loss_delta, loss_abs, loss_intent, w_intent= self.model.loss(
                    pred_deltas=outputs_deltas,
                    y_truth_abs=batch_y,
                    x_enc=batch_x,
                    mask_y=mask_y,
                    intent_vectors=intent_vectors,
                )
                
                total_loss_list.append(loss_delta.item())
                total_loss_absolute.append(loss_abs.item())
                total_loss_intent.append(loss_intent.item())
                total_loss_delta.append(loss_delta.item())
        
        total_loss = np.average(total_loss_list)
        total_loss_absolute = np.average(total_loss_absolute)
        total_loss_intent = np.average(total_loss_intent)
        total_loss_delta = np.average(total_loss_delta)
        self.model.train()
        return total_loss, total_loss_delta, total_loss_absolute, total_loss_intent
    def train(self, setting):
        """
        训练模型
        [已修改] "setting" 现在是 "Experiment/Run" 路径
        """
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')
        
        # [修改] path 现在是唯一的 "运行" 路径
        # setting = "Experiment_Name/Run_Name"
        # path = "./checkpoints/Experiment_Name/Run_Name"
        ckpt_path = os.path.join(setting, 'checkpoints')
        if not os.path.exists(ckpt_path):
            os.makedirs(ckpt_path, exist_ok=True)
        
        time_now = time.time()
        
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        model_optim = self._select_optimizer()
        
        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()
        
        for epoch in range(self.args.train_epochs):
            # ... (epoch 循环内部逻辑保持不变) ...
            iter_count = 0
            train_total_loss = []
            train_loss_delta = []
            train_loss_absolute = []
            train_loss_intent = []

            
            self.model.train()
            epoch_time = time.time()

            if epoch < 5:
                self.model.sampling_prob = 1.0  # 前10轮，强制使用真值 COG，教模型“什么样的航向对应哪条路”
            elif epoch < 10:
                self.model.sampling_prob = 0.8  # 逐渐引入自身预测
            else:
                self.model.sampling_prob = 0.5  # 后期模拟真实推理环境
            
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
                        
                        # [修正] 应当在混合损失上反向传播
                        back_loss = loss + 0.5 * loss_absolute # (假设 lambda=0.5)
                        
                        train_loss.append(loss.item())
                        train_loss_absolute.append(loss_absolute.item())
                else:
                    pred_deltas, intent_vectors = self.model(
                        x_enc=batch_x,
                        y_truth_abs=batch_y[..., :2],
                        mask_x=mask_x,
                        mask_y=mask_y,
                        A_social_t=A_social.to(self.device),
                        edge_features=edge_features.to(self.device),
                    )
                    
                    total_loss, loss_delta, loss_abs, loss_intent, w_intent = self.model.loss(
                        pred_deltas=pred_deltas,
                        y_truth_abs=batch_y,
                        x_enc=batch_x,
                        mask_y=mask_y,
                        intent_vectors=intent_vectors,
                    )
                    
                    train_total_loss.append(total_loss.item())
                    train_loss_delta.append(loss_delta.item())
                    train_loss_absolute.append(loss_abs.item())
                    train_loss_intent.append(loss_intent.item())
                
                if (i + 1) % 100 == 0:
                    
                    # print(f"\titers: {i + 1}, epoch: {epoch + 1} | loss delta: {loss.item():.7f},  loss absolute: {loss_absolute.item():.7f}, motion loss: {loss_motion.item():.7f} ")

                    print(f"\titers: {i + 1}, epoch: {epoch + 1} | loss total: {total_loss.item():.7f},  loss delta: {loss_delta.item():.7f},  loss absolute: {loss_abs.item():.7f}, intent loss: {loss_intent.item():.7f}")
                    
                    # ... (speed, left_time) ...
                
                if self.args.use_amp:
                    scaler.scale(back_loss).backward() # [修正]
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    
                    # back_loss = loss + 0.5 * loss_absolute + 0.5 * loss_motion # [修正]
                    back_loss = total_loss # [修正]
                    back_loss.backward()
                    
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    model_optim.step()
            
            # ... (Epoch 总结) ...
            print(f"Epoch: {epoch + 1} cost time: {time.time() - epoch_time}")
            train_total_loss = np.average(train_total_loss)
            train_loss_delta = np.average(train_loss_delta)
            train_loss_absolute = np.average(train_loss_absolute)
            train_loss_intent = np.average(train_loss_intent)

            
            # [修正] vali_loss 现在是 absolute loss
            # vali_loss, vali_loss_absolute, vali_loss_motion = self.vali(vali_loader)
            # test_loss, test_loss_absolute, test_loss_motion = self.vali(test_loader) # (假设 test 也用 vali 逻辑)

            # [修正] vali_loss 现在是 absolute loss
            vali_loss, vali_loss_delta, vali_loss_absolute, vali_loss_intent = self.vali(vali_loader)
            test_loss, test_loss_delta, test_loss_absolute, test_loss_intent = self.vali(test_loader) # (假设 test 也用 vali 逻辑)
            

            # print(f"Epoch: {epoch + 1}, Steps: {train_steps} | Train Loss: {train_loss:.7f}, {train_loss_absolute:.7f}, {train_loss_motion:.7f} Vali Loss: {vali_loss:.7f}, {vali_loss_absolute:.7f}, {vali_loss_motion:.7f} Test Loss: {test_loss:.7f}, {test_loss_absolute:.7f}, {test_loss_motion:.7f}")

            print(f"Epoch: {epoch + 1}, Steps: {train_steps} | Train total Loss: {train_total_loss:.7f}, Train delta loss {train_loss_delta:.7f}, Train abs loss {train_loss_absolute:.7f}, Train intent loss {train_loss_intent:.7f}")
            print(f"                                           Vali total Loss: {vali_loss:.7f}, Vali delta loss {vali_loss_delta:.7f}, Vali abs loss {vali_loss_absolute:.7f}, Vali intent loss {vali_loss_intent:.7f}")
            print(f"                                           Test total Loss: {test_loss:.7f}, Test delta loss {test_loss_delta:.7f}, Test abs loss {test_loss_absolute:.7f}, Test intent loss {test_loss_intent:.7f}")
            # early_stopping 监控的是 vali_loss_absolute
            early_stopping(vali_loss_absolute, self.model, ckpt_path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            
            adjust_learning_rate(model_optim, epoch + 1, self.args)
        
        # [修改] 加载最佳模型 (从 "run" 路径加载)
        best_model_path = os.path.join(ckpt_path, 'checkpoint.pth')
        self.model.load_state_dict(torch.load(best_model_path))
        
        return self.model

    def test(self, setting, test=0):
        """
        测试模型
        [已修改] "setting" 现在是 "Experiment/Run" 路径
        """
        test_data, test_loader = self._get_data(flag='test')
        
        # [修改] 结果路径 (保存到 ./results/...)

        results_path = os.path.join(setting, 'results')
        if not os.path.exists(results_path):
            os.makedirs(results_path, exist_ok=True)
        if test:
            print('loading model')
            # [修改] 检查点路径 (加载从 ./checkpoints/...)
            checkpoint_path = os.path.join(setting, 'checkpoints', 'checkpoint.pth')
            self.model.load_state_dict(torch.load(checkpoint_path))
        hists = []
        preds = []
        trues = []
        masks_list = []
        
        # [移除了 'folder_path' 的旧定义]
        
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, mask_x, mask_y, ship_count, _, A_social, edge_features) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                # [修改] 确保 mask 和 y_truth_abs=None 被传递
                outputs_deltas = self.model(
                    x_enc=batch_x,
                    y_truth_abs=None, # 触发推理
                    mask_x=mask_x.to(self.device),
                    mask_y=mask_y.to(self.device),
                    A_social_t=A_social.to(self.device),
                    edge_features=edge_features.to(self.device),

                )
                
                outputs_absolute = self.model.integrate(outputs_deltas, batch_x)
                
                hists.append(batch_x[..., :2].detach().cpu().numpy())
                preds.append(outputs_absolute.detach().cpu().numpy())
                trues.append(batch_y[..., :2].detach().cpu().numpy())
                masks_list.append(mask_y.detach().cpu().numpy())
        
        scaler_file = os.path.join(self.args.root_path, 'scaler_params.npy')
        print('Loading scaler parameters from:', scaler_file)
        scaler_params = np.load(scaler_file, allow_pickle=True).item()
        mean, std = scaler_params['mean'], scaler_params['std']



        # ... (拼接 preds, trues, masks 不变) ...
        hists = np.concatenate(hists, axis=0)
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        masks = np.concatenate(masks_list, axis=0).astype(bool)

        print('hists shape:', hists.shape)
        print('preds shape:', preds.shape)
        print('trues shape:', trues.shape)

        # 反标准化
        hists_invers = hists * std[:2] + mean[:2]
        preds_invers = preds * std[:2] + mean[:2]
        trues_invers = trues * std[:2] + mean[:2]
        
        print('test shape:', preds.shape, trues.shape, masks.shape)
        
        # [移除了 'folder_path' 的旧定义]

        # ... (计算 ADE, FDE, metric 不变) ...
        ade = ADE(preds, trues, mask=masks)
        fde = FDE(preds, trues, mask=masks)
        mae, mse, rmse, mape, mspe = metric(preds, trues, mask=masks)
        
        print(f'Metrics (on valid data only):')
        print(f'mse:{mse:.4f}, mae:{mae:.4f}, rmse:{rmse:.4f}, mape:{mape:.4f}, mspe:{mspe:.4f}')
        print(f'ADE:{ade:.4f}, FDE:{fde:.4f}')
        
        # [修改] 保存结果到新的 "results_path"
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
        """
        预测（用于实际部署）
        [已修改] "setting" 现在是 "Experiment/Run" 路径
        """
        pred_data, pred_loader = self._get_data(flag='test')
        
        # [修改] 结果路径 (保存到 ./results/...)
        results_path = os.path.join('./results/', setting)
        if not os.path.exists(results_path):
            os.makedirs(results_path, exist_ok=True)
            
        if load:
            # [修改] 检查点路径 (加载从 ./checkpoints/...)
            checkpoint_path = os.path.join(self.args.checkpoints, setting, 'checkpoint.pth')
            self.model.load_state_dict(torch.load(checkpoint_path))
        
        preds = []
        
        self.model.eval()
        with torch.no_grad():
            for i,(batch_x, batch_y, mask_x, mask_y, ship_count, _) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                
                # [修改] 确保 mask 和 y_truth_abs=None 被传递
                outputs_deltas = self.model(
                    x_enc=batch_x,
                    y_truth_abs=None,
                    mask_x=mask_x.to(self.device),
                    mask_y=mask_y.to(self.device)
                )
                
                # [修改] 将增量重建为绝对坐标
                outputs_absolute = self.model.integrate(outputs_deltas[..., :2], batch_x)
                
                preds.append(outputs_absolute.detach().cpu().numpy())
        
        preds = np.concatenate(preds, axis=0)
        
        # [移除了 'folder_path' 的旧定义]
        
        # [修改] 保存预测结果到 "results_path"
        np.save(os.path.join(results_path, 'real_prediction.npy'), preds)
        
        return preds