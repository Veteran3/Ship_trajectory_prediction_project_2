import yaml
import argparse
import os

class Args:
    """
    创建一个类似 argparse.Namespace 的对象，方便属性访问 (args.model)
    """
    def __init__(self, **entries):
        self.__dict__.update(entries)

def get_args_from_yaml(config_path='config.yaml'):
    """
    从 YAML 文件加载参数，并允许命令行参数覆盖 YAML 配置
    """
    # 1. 首先读取 YAML 文件
    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)

    # 2. 特殊处理：root_path 通常依赖 data 字段动态生成
    if 'root_path' not in config_dict:
        data_name = config_dict.get('data', '30s')
        config_dict['root_path'] = f'./data/{data_name}/'

    # 3. 设置 argparse，允许命令行覆盖 YAML 中的任何参数
    parser = argparse.ArgumentParser(description='Ship Trajectory Prediction (YAML Config)')
    
    # 必须参数：配置文件路径（可选）
    parser.add_argument('--config', type=str, default=config_path, help='Path to the YAML config file')

    # 动态添加 YAML 中的所有键作为命令行参数
    # 这样你依然可以使用 python main.py --gpu 1 --batch_size 64 来临时覆盖配置
    for key, value in config_dict.items():
        # 根据 value 的类型自动推断 type
        arg_type = type(value) if value is not None else str
        # 对于 bool 类型，yaml 读取出来是 bool，但 argparse 最好用 int (0/1) 或者特殊的 bool 处理
        # 你原本的代码用的是 int (0/1) 代表 bool，这里保持一致
        parser.add_argument(f'--{key}', type=arg_type, default=value)

    # 补充一些原本 argparse 中有逻辑处理的参数
    if 'device_ids' not in config_dict:
        parser.add_argument('--device_ids', type=list, default=[])

    # 解析命令行参数
    # parse_args 会使用 default=value (来自 YAML)，如果命令行指定了 --key，则使用命令行指
    args_namespace = parser.parse_args()

    # 4. 后处理逻辑 (原 get_args 中的逻辑)
    if args_namespace.use_multi_gpu and args_namespace.use_gpu:
        if isinstance(args_namespace.devices, str):
            args_namespace.device_ids = [int(id_) for id_ in args_namespace.devices.split(',')]
            args_namespace.gpu = args_namespace.device_ids[0]
    
    return args_namespace

# ================= 使用示例 =================
if __name__ == '__main__':
    # 假设你的 yaml 文件叫 config.yaml
    try:
        args = get_args_from_yaml(r'/mnt/stu/ZhangDong/2_PhD_projects/0_1_My_model/configs/config.yaml')
        print("="*40)
        print("Loaded Configuration:")
        for k, v in vars(args).items():
            print(f"{k}: {v}")
        print("="*40)
    except FileNotFoundError:
        print("错误：未找到 config.yaml 文件，请确保文件存在。")