import argparse
import os
import torch
import random
import numpy as np
from exp.exp_forecasting_V2 import Exp_Forecasting
import time

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


def get_args():
    """
    获取命令行参数
    """
    parser = argparse.ArgumentParser(description='Ship Trajectory Prediction with Transformer')
    
    # ==================== 基本配置 ====================
    parser.add_argument('--task_name', type=str, default='ship_trajectory_forecast',
                        help='task name')
    parser.add_argument('--is_training', type=int, default=0,
                        help='status: 1 for training, 0 for testing')
    parser.add_argument('--model_id', type=str, default='ship_traj',
                        help='model id')
    parser.add_argument('--model', type=str, default='V2_2_ASTGNN',
                        help='model name')
    
    # ==================== 数据配置 ====================
    parser.add_argument('--root_path', type=str, default='./data/',
                        help='root path of the data file')
    parser.add_argument('--train_data_path', type=str, default='train.npz',
                        help='train data file')
    parser.add_argument('--val_data_path', type=str, default='val.npz',
                        help='validation data file')
    parser.add_argument('--test_data_path', type=str, default='test.npz',
                        help='test data file')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/',
                        help='location of model checkpoints')
    
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
    parser.add_argument('--d_model', type=int, default=64,
                        help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8,
                        help='number of attention heads')
    parser.add_argument('--e_layers', type=int, default=4,
                        help='number of encoder layers')
    parser.add_argument('--d_layers', type=int, default=4,
                        help='number of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048,
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
    
    args = parser.parse_args()
    
    # 处理多GPU设置
    if args.use_multi_gpu and args.use_gpu:
        args.device_ids = [int(id_) for id_ in args.devices.split(',')]
        args.gpu = args.device_ids[0]
    
    return args


def main():
    """
    主函数
    """
    # 获取参数
    args = get_args()
    
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
    setting = '{}_{}_sl{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_{}_{}_2'.format(
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
        args.seed
    )
    
    print(f'\nExperiment Setting: {setting}\n')
    
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


if __name__ == '__main__':
    main()