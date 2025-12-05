import argparse
import os
import torch
import random
import numpy as np
from exp.exp_forecasting_VV1 import Exp_Forecasting
import time
import json
import traceback
import sys
import yaml  # 需要 pip install pyyaml

# ==================== 日志工具 ====================
class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

# ==================== 随机种子 ====================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_args():
    """
    获取命令行参数
    """
    parser = argparse.ArgumentParser(description='Resume Training')
    
    # 核心：指定接续的路径
    parser.add_argument('--resume_path', type=str, required=True,
                        help='The path to the experiment directory to resume')
    
    # 指定配置文件（用于补充新参数）
    parser.add_argument('--config', type=str, default=r'./configs/config.yaml',
                        help='Path to the config file (to fill missing args)')
    
    # 允许覆盖 GPU 设置
    parser.add_argument('--use_gpu', type=int, default=1, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
    parser.add_argument('--use_multi_gpu', type=int, default=0, help='use multiple gpus')
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids')

    return parser.parse_args()

def main():
    # 1. 获取命令行参数
    cmd_args = get_args()
    setting = cmd_args.resume_path

    if not os.path.exists(setting):
        raise FileNotFoundError(f"指定的接续路径不存在: {setting}")

    print('=' * 80)
    print(f'RESUMING TRAINING FROM: {setting}')
    print('=' * 80)

    # 2. 寻找 checkpoint
    ckpt_dir = os.path.join(setting, 'checkpoints')
    latest_ckpt = os.path.join(ckpt_dir, 'checkpoint.pth')
    best_ckpt = os.path.join(ckpt_dir, 'checkpoint.pth')
    
    # 优先用 latest (包含最新的优化器状态)，否则用 best
    target_ckpt = latest_ckpt if os.path.exists(latest_ckpt) else best_ckpt
    
    if not os.path.exists(target_ckpt):
        raise FileNotFoundError(f"在 {ckpt_dir} 中未找到 checkpoint 文件！")

    print(f"Loading checkpoint from: {target_ckpt}")
    
    # 3. 加载 Checkpoint 中的 args
    checkpoint = torch.load(target_ckpt, map_location='cpu', weights_only=False)
    if 'args' not in checkpoint:
        raise ValueError("Checkpoint 中未找到 'args' 字段！")
    
    args = checkpoint['args']

    # -------------------------------------------------------------------------
    # 4. [关键修改] 读取 YAML 并补充缺失参数
    # -------------------------------------------------------------------------
    if os.path.exists(cmd_args.config):
        print(f"Checking {cmd_args.config} for new parameters...")
        with open(cmd_args.config, 'r', encoding='utf-8') as f:
            yaml_config = yaml.safe_load(f)
        
        # 遍历 YAML 中的键值对
        for key, value in yaml_config.items():
            # 如果存档的 args 里没有这个 key（说明是新代码加的参数），则添加进去
            if not hasattr(args, key):
                print(f"  [Auto-Fill] Adding new parameter from YAML: {key} = {value}")
                setattr(args, key, value)
    else:
        print(f"Warning: Config file {cmd_args.config} not found. Using only checkpoint args.")
    # -------------------------------------------------------------------------

    # 5. 强制覆盖设置
    args.resume = 1         # 开启接续
    args.is_training = 1    # 确保训练模式
    
    # 覆盖 GPU 设置 (使用当前命令行的输入)
    args.use_gpu = cmd_args.use_gpu
    args.gpu = cmd_args.gpu
    args.use_multi_gpu = cmd_args.use_multi_gpu
    args.devices = cmd_args.devices
    
    # 处理多GPU ID
    if args.use_multi_gpu and args.use_gpu:
        args.device_ids = [int(id_) for id_ in args.devices.split(',')]
        args.gpu = args.device_ids[0]

    # 6. 设置环境
    set_seed(args.seed)

    # 设置 Logger
    log_file_path = os.path.join(setting, 'run_log.txt')
    original_stdout = sys.stdout
    sys.stdout = Logger(log_file_path)

    try:
        # 7. 开始实验
        exp = Exp_Forecasting(args)

        print('\n' + '>' * 80)
        print('Resuming Training Stage')
        print('>' * 80)

        exp.train(setting)

        print('\n' + '>' * 80)
        print('Testing Stage (After Resume)')
        print('>' * 80)
        exp.test(setting, test=1)

        if args.use_gpu:
            torch.cuda.empty_cache()

        print('\n' + '=' * 80)
        print('Resumed Experiment Finished!')
        print('=' * 80)

    except KeyboardInterrupt:
        if isinstance(sys.stdout, Logger):
            sys.stdout.log.close()
        sys.stdout = original_stdout
        print('\nUser interrupted.')

    except Exception as e:
        if isinstance(sys.stdout, Logger):
            sys.stdout.log.close()
        sys.stdout = original_stdout
        print('\nException occurred!')
        traceback.print_exc()
        raise e

if __name__ == '__main__':
    main()