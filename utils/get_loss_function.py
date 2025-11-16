import torch
from torch import nn

def get_loss_function(loss_name):
    """
    获取损失函数
    """
    if loss_name == 'mse':
        return nn.MSELoss()
    elif loss_name == 'mae':
        return nn.L1Loss()
    elif loss_name == 'huber':
        return nn.SmoothL1Loss()
    else:
        raise ValueError(f"Unsupported loss function: {loss_name}")