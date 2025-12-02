import argparse
import torch
import os
import warnings
from exp.exp_forecasting_V4 import Exp_Forecasting # 确保从正确的位置导入

# 忽略警告
warnings.filterwarnings('ignore')

def main():
    # --- 1. 参数解析 ---
    # 我们必须重新定义所有必要的参数，以便模型能被正确构建
    parser = argparse.ArgumentParser(description='[ASTGNN] Trajectory Forecasting - Testing')

    # [关键] 这是您要运行的实验的唯一标识符
    parser.add_argument('--setting', type=str, required=True, 
                        help='The "setting" string, e.g., "ship_traj_V2_2.../run_seed_42_..."')
    
    # --- 模型参数 (必须与训练时一致) ---
    # ==================== 基本配置 ====================
    parser.add_argument('--task_name', type=str, default='ship_trajectory_forecast',
                        help='task name')
    parser.add_argument('--is_training', type=int, default=0,
                        help='status: 1 for training, 0 for testing')
    parser.add_argument('--model_id', type=str, default='ship_traj',
                        help='model id')
    parser.add_argument('--model', type=str, default='V3_0_0_ASTGNN',
                        help='model name')
    
    # ==================== 数据配置 ====================
    parser.add_argument('--root_path', type=str, default='./data/30s',
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
    # parser.add_argument('--lane_table_path', type=str, default=None,
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

    args = parser.parse_args()

    # --- 2. 设置设备 ---
    if args.use_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu) if not args.use_multi_gpu else args.device_ids
        device = torch.device('cuda:{}'.format(args.gpu))
        print(f'Using GPU: {device}')
    else:
        device = torch.device('cpu')
        print('Using CPU')
    
    args.device = device
    
    # --- 3. 初始化实验 ---
    # 这将自动:
    # 1. 创建 Exp_Forecasting 实例
    # 2. 调用 exp._build_model() 来创建模型骨架
    exp = Exp_Forecasting(args)

    # --- 4. 运行测试 ---
    print(f'>>> 开始测试: {args.setting} <<<<<<<<<<<<<<<<<<<<<<')
    
    # 我们调用 exp.test(), 传入 setting 字符串和 test=1
    # test=1 会触发您 test() 方法内部的 "loading model" 逻辑
    exp.test(args.setting, test=1)

if __name__ == '__main__':
    main()
    """
    python test.py \
    --model V3_1_0_ASTGNN \
    --setting "/mnt/stu/ZhangDong/2_PhD_projects/0_0_My_model/experiments/ship_traj_V3_1_0_ASTGNN_sl8_pl12_dm64_nh8_el4_dl4_df256_Exp_30s/run_seed2024_20251202013920" \
    --seq_len 8 \
    --pred_len 12 \
    --d_model 64 \
    --n_heads 8 \
    --e_layers 4 \
    --d_layers 4 \
    --d_ff 2048 \
    --num_ships 17 \
    --use_gpu 1
    
    """