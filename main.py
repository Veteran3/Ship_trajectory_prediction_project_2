import argparse
import os
import torch
import random
import numpy as np
from exp.exp_forecasting_V3_1 import Exp_Forecasting
import time
import json
import shutil     # [新] 用于删除文件夹
import traceback  # [新] 用于打印报错信息

import sys
import os
# ... (其他 import)

class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout  # 记录原始的终端输出句柄
        self.log = open(filename, "a", encoding='utf-8') # 打开日志文件 (追加模式)

    def write(self, message):
        self.terminal.write(message) # 写到屏幕
        self.log.write(message)      # 写到文件
        self.log.flush()             # 立即刷新缓冲区，防止程序崩了没保存

    def flush(self):
        # 这个方法是必须的，为了兼容 Python 的 IO 接口
        self.terminal.flush()
        self.log.flush()

def set_seed(seed):
    """
    设置随机种子，保证实验可复现
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_args(data):
    """
    获取命令行参数
    """
    parser = argparse.ArgumentParser(description='Ship Trajectory Prediction with Transformer')
    
    experiment_desc = """
    
## 实验目的：
v3.1.0.1

Loss = loss_delta * 0.5  + loss_absolute 

"""
    # ==================== 基本配置 ====================
    parser.add_argument('--task_name', type=str, default='ship_trajectory_forecast',
                        help='task name')
    parser.add_argument('--is_training', type=int, default=1,
                        help='status: 1 for training, 0 for testing')
    parser.add_argument('--model_id', type=str, default='ship_traj',
                        help='model id')
    parser.add_argument('--model', type=str, default='V3_1_0_ASTGNN',
                        help='model name')
    
    # ==================== 数据配置 ====================
    parser.add_argument('--root_path', type=str, default=f'./data/{data}/',
                        help='root path of the data file')
    parser.add_argument('--train_data_path', type=str, default='train.npz',
                        help='train data file')
    parser.add_argument('--val_data_path', type=str, default='val.npz',
                        help='validation data file')
    parser.add_argument('--test_data_path', type=str, default='test.npz',
                        help='test data file')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/',
                        help='location of model checkpoints')
    parser.add_argument('--lane_table_path', type=str, default=r'/mnt/stu/ZhangDong/2_PhD_projects/0_0_My_model/data/lane_table_with_next_lane.csv',
                        help='path to the lane table file')
    
    # ==================== 数据预处理 ====================
    parser.add_argument('--seq_len', type=int, default=8,
                        help='input sequence length')
    parser.add_argument('--pred_len', type=int, default=12,
                        help='prediction sequence length')
    parser.add_argument('--num_ships', type=int, default=17,
                        help='number of ships per frame')
    parser.add_argument('--num_features', type=int, default=4,
                        help='number of features (lon, lat, cog, sog)')
    parser.add_argument('--scale', type=int, default=1,
                        help='whether to scale data')
    parser.add_argument('--scale_type', type=str, default='standard',
                        choices=['standard', 'minmax'],
                        help='normalization type: standard or minmax')
    parser.add_argument('--predict_position_only', type=int, default=1,
                        help='only predict position (lon, lat), ignore COG and SOG')
    
    # ==================== 模型配置 ====================
    parser.add_argument('--in_feature', type=int, default=7,
                        help='input feature dimension')
    parser.add_argument('--d_model', type=int, default=64,
                        help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8,
                        help='number of attention heads')
    parser.add_argument('--e_layers', type=int, default=4,
                        help='number of encoder layers')
    parser.add_argument('--d_layers', type=int, default=4,
                        help='number of decoder layers')
    parser.add_argument('--d_ff', type=int, default=256,
                        help='dimension of feed-forward network')
    parser.add_argument('--factor', type=int, default=5,
                        help='attention factor')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='dropout rate')
    parser.add_argument('--activation', type=str, default='gelu',
                        choices=['relu', 'gelu'],
                        help='activation function')
    parser.add_argument('--output_attention', type=int, default=0,
                        help='whether to output attention weights')
    
    # ==================== 训练配置 ====================
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch size of train input data')
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                        help='initial learning rate')
    parser.add_argument('--loss', type=str, default='mse',
                        choices=['mse', 'mae', 'huber'],
                        help='loss function')
    parser.add_argument('--train_epochs', type=int, default=100,
                        help='number of training epochs')
    parser.add_argument('--patience', type=int, default=10,
                        help='early stopping patience')
    parser.add_argument('--lradj', type=str, default='type1',
                        choices=['type1', 'type2', 'cosine'],
                        help='learning rate adjustment strategy')
    
    # ==================== GPU配置 ====================
    parser.add_argument('--use_gpu', type=int, default=1,
                        help='use gpu')
    parser.add_argument('--gpu_type', type=str, default='cuda',
                        choices=['cuda', 'mps'],
                        help='gpu type: cuda or mps (for Mac)')
    parser.add_argument('--gpu', type=int, default=0,
                        help='gpu device id')
    parser.add_argument('--use_multi_gpu', type=int, default=0,
                        help='use multiple gpus')
    parser.add_argument('--devices', type=str, default='0,1,2,3',
                        help='device ids of multiple gpus')
    
    # ==================== 其他配置 ====================
    parser.add_argument('--num_workers', type=int, default=4,
                        help='data loader num workers')
    parser.add_argument('--use_amp', type=int, default=0,
                        help='use automatic mixed precision training')
    parser.add_argument('--seed', type=int, default=2024,
                        help='random seed')
    parser.add_argument('--des', type=str, default='Exp',
                        help='experiment description')
    
    # ==================== 采样概率配置 ====================
    parser.add_argument('--ANNEAL_START_EPOCH', type=int, default=1,
                        help='epoch to start annealing sampling probability')
    parser.add_argument('--ANNEAL_END_EPOCH', type=int, default=50,
                        help='epoch to end annealing sampling probability')
    parser.add_argument('--PROB_START', type=float, default=1.0,
                        help='initial sampling probability')
    parser.add_argument('--PROB_END', type=float, default=0.2,
                        help='final sampling probability')
    
    # ==================== 语义影响力配置 ====================
    parser.add_argument('--social_sigma', type=float, default=1.0,
                        help='sigma value for semantic social influence')
    parser.add_argument('--tcpa_thresh', type=float, default=300.0,
                        help='threshold value for TCPA')
    parser.add_argument('--dcpa_thresh', type=float, default=1000.0,
                        help='threshold value for DCPA')
    
    args = parser.parse_args()
    
    # 处理多GPU设置
    if args.use_multi_gpu and args.use_gpu:
        args.device_ids = [int(id_) for id_ in args.devices.split(',')]
        args.gpu = args.device_ids[0]
    
    return args, experiment_desc


def main():
    """
    主函数
    """
    data = '30s'
    # 获取时间戳
    timestamp = time.strftime('%Y%m%d%H%M%S', time.localtime())
    
    # 获取参数
    args, experiment_desc = get_args(data)
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 打印配置
    print('=' * 80)
    print('Ship Trajectory Prediction with Transformer')
    print('=' * 80)
    print('Args in experiment:')
    for arg in vars(args):
        print(f'  {arg}: {getattr(args, arg)}')
    print('=' * 80)
    
    # 构建实验设置名称
    experiment_name = '{}_{}_sl{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_{}_{}'.format(
        args.model_id,
        args.model,
        args.seq_len,
        args.pred_len,
        args.d_model,
        args.n_heads,
        args.e_layers,
        args.d_layers,
        args.d_ff,
        args.des,
        data
    )
    run_name = f"run_seed{args.seed}_{timestamp}"

    

    description_content = f"""
# 实验运行描述 (Experiment Run Description)

* **实验 (Experiment):** `{experiment_name}`
* **运行 (Run):** `{run_name}`
* **时间 (Timestamp):** `{timestamp}`
* **种子 (Seed):** `{args.seed}`

---

## 运行备注 (Description)

{experiment_desc.strip()}

---

## 全部参数 (All Arguments)


{json.dumps(vars(args), indent=4, ensure_ascii=False)}
"""
    setting = os.path.join('experiments/', experiment_name + '/' + run_name)
   
    # 写入文件 (使用 utf-8 编码来支持中文)
    
    try:
        if not os.path.exists(setting):
            os.makedirs(setting, exist_ok=True)
            is_newly_created = True
        desc_path = os.path.join(setting, 'description.md')
        try:
            with open(desc_path, 'w', encoding='utf-8') as f:
                f.write(description_content)
            print(f"实验描述已保存到: {desc_path}")
        except Exception as e:
            print(f"警告: 保存实验描述失败 - {e}")
            print(f'\nExperiment Setting: {setting}\n')
        
        log_file_path = os.path.join(setting, 'run_log.txt')
        original_stdout = sys.stdout
        sys.stdout = Logger(log_file_path)

        # 创建实验对象
        exp = Exp_Forecasting(args)
        
        # 训练模式
        if args.is_training:
            print('>' * 80)
            print('Training Stage')
            print('>' * 80)
            
            # 训练
            exp.train(setting)
            
            # 测试
            print('\n' + '>' * 80)
            print('Testing Stage')
            print('>' * 80)
            exp.test(setting, test=1)
            
            # 清理GPU缓存
            if args.use_gpu:
                torch.cuda.empty_cache()
        
        # 测试模式
        else:
            print('>' * 80)
            print('Testing Stage')
            print('>' * 80)
            exp.test(setting, test=1)
            
            # 清理GPU缓存
            if args.use_gpu:
                torch.cuda.empty_cache()
        
        print('\n' + '=' * 80)
        print('Experiment Finished!')
        print('=' * 80)


    # =========================================================
    # 分支 1: 用户手动停止 (Ctrl+C) -> 保留文件夹
    # =========================================================
    except KeyboardInterrupt:
        # A. 恢复控制台，关闭日志文件 (防止文件损坏)
        if isinstance(sys.stdout, Logger):
            sys.stdout.log.close()
        sys.stdout = original_stdout

        print('\n' + '='*30)
        print("用户手动终止程序 (KeyboardInterrupt)")
        print(f"结果已保留在: {setting}")
        print('='*30)
        # 这里不执行删除操作，直接退出

    # =========================================================
    # 分支 2: 程序异常崩溃 (Bug/Error) -> 删除文件夹
    # =========================================================
    except Exception as e:
        # A. 恢复控制台 (必须先做，否则无法删除文件)
        if isinstance(sys.stdout, Logger):
            sys.stdout.log.close()
        sys.stdout = original_stdout

        print('\n' + '='*30)
        print("检测到程序异常崩溃！开始清理...")
        print('='*30)
        
        # B. 打印报错信息 (非常重要，否则不知道错哪了)
        traceback.print_exc()

        # C. 执行删除
        if is_newly_created and os.path.exists(setting):
            try:
                print(f"\n正在自动清理失败的运行目录: {setting} ...")
                shutil.rmtree(setting)
                print(">> 清理完成 (垃圾文件已删除)。")
            except OSError as err:
                print(f">> 清理失败: {err}")
        
        # D. 抛出异常，终止程序
        raise e

    


if __name__ == '__main__':
    main()