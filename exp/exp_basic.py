import os
import torch
# from models import Transformer_V1_0, Transformer_V1_1
from torch.utils.data import DataLoader
from models import  V3_0_0_ASTGNN, V3_1_0_ASTGNN, V3_1_0_1_ASTGNN


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
          
            'V3_0_0_ASTGNN': V3_0_0_ASTGNN,
            'V3_1_0_ASTGNN': V3_1_0_ASTGNN,
            'V3_1_0_1_ASTGNN': V3_1_0_1_ASTGNN,

        }
       

        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu and self.args.gpu_type == 'cuda':
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        elif self.args.use_gpu and self.args.gpu_type == 'mps':
            device = torch.device('mps')
            print('Use GPU: mps')
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self, flag):
        # ... (此方法保持不变) ...
        args = self.args

        if  args.model in ['V3_0_0_ASTGNN', 'V3_1_0_ASTGNN', 'V3_1_1_ASTGNN', 'V3_1_2_ASTGNN', 'V3_1_0_1_ASTGNN']:
            from data_provider.data_loader_5_0 import ShipTrajectoryDataset

        else:
            raise ValueError('Please select the correct data_loader version for the model!')
        data_dict = {
            'train': (args.train_data_path, True),
            'val': (args.val_data_path, True),
            'test': (args.test_data_path, False)
        }
        data_path, shuffle_flag = data_dict[flag]
        data_set = ShipTrajectoryDataset(
            root_path=args.root_path,
            data_path=data_path,
            flag=flag,
            size=[args.seq_len, args.pred_len],
            num_ships=args.num_ships,
            num_features=args.num_features,
            scale=args.scale,
            lane_table_path=args.lane_table_path,
            predict_position_only=args.predict_position_only,
            scale_type=args.scale_type
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

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
