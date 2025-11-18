import os
import torch
# from models import Transformer_V1_0, Transformer_V1_1

from models import  V1_0_Transformer, V1_1_Transformer, V2_0_ASTGNN, V2_1_ASTGNN, V2_2_ASTGNN, V2_2_1_ASTGNN

class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            # 'Transformer_V1': Transformer_V1_0,
            'Transformer_V1_1': V1_1_Transformer,
            'V2_0_ASTGNN': V2_0_ASTGNN,
            'V2_1_ASTGNN': V2_1_ASTGNN,
            'V2_2_ASTGNN': V2_2_ASTGNN,
            'V2_2_1_ASTGNN': V2_2_1_ASTGNN,
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

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
