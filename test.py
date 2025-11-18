import argparse
import torch
import os
import warnings
from exp.exp_forecasting import Exp_Forecasting # 确保从正确的位置导入

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
    parser.add_argument('--model', type=str, default='ASTGNN_Ship',
                        help='Model name (e.g., ASTGNN_Ship)')
    parser.add_argument('--d_model', type=int, default=64, help='Dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='Number of heads')
    parser.add_argument('--e_layers', type=int, default=4, help='Number of encoder layers')
    parser.add_argument('--d_layers', type=int, default=4, help='Number of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='Dimension of FFN')
    
    # --- 数据参数 (必须与训练时一致) ---
    parser.add_argument('--root_path', type=str, default='./data/', help='Root path of data file')
    parser.add_argument('--test_data_path', type=str, default='test.npy', help='Test data file name')
    parser.add_argument('--seq_len', type=int, default=8, help='Input sequence length')
    parser.add_argument('--pred_len', type=int, default=12, help='Prediction sequence length')
    parser.add_argument('--num_ships', type=int, default=17, help='Number of ships (nodes)')
    parser.add_argument('--num_features', type=int, default=4, help='Number of input features')
    parser.add_argument('--scale', type=bool, default=True, help='Whether to scale data')
    
    # (添加您在模型中使用的任何其他自定义参数)
    # parser.add_argument('--static_feature_dim', type=int, default=5, help='Dimension of static features')
    parser.add_argument('--loss_fn', type=str, default='mse', help='Loss function')
    
    # --- 运行参数 ---
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for testing')
    parser.add_argument('--num_workers', type=int, default=4, help='Data loader num workers')
    parser.add_argument('--use_gpu', type=bool, default=True, help='Use GPU')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID')
    parser.add_argument('--use_multi_gpu', action='store_true', help='Use multiple GPUs')
    parser.add_argument('--device_ids', type=str, default='0,1', help='Device IDs for multi-GPU')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='Checkpoint save location')

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
    --model ASTGNN_Ship \
    --setting "ship_traj_V2_2_ASTGNN_sl8_pl12_..._Exp_2024/run_seed_42_..." \
    --seq_len 8 \
    --pred_len 12 \
    --d_model 64 \
    --n_heads 8 \
    --e_layers 4 \
    --d_layers 4 \
    --d_ff 2048 \
    --num_ships 17 \
    --num_features 4 \
    --root_path ./data/ \
    --test_data_path test.npy \
    --use_gpu True
    
    """