import os
import torch
from typing import Any
import time


def load_epoch(model_path: str, epoch: int) -> Any:
    print('loading from epoch.%04d.pth' % epoch)
    return torch.load(os.path.join(model_path, 'epoch.%04d.pth' % epoch),
                      map_location='cpu')


def load_best_epoch(model_path: str) -> Any:
    print('loading from best.pth')
    return torch.load(os.path.join(model_path, 'best.pth'),
                      map_location='cpu')


### Define hook functions ###
take_time_dict = {}


def take_time_pre(layer_name, module, input):
    take_time_dict[layer_name] = time.time()


def take_time(layer_name, module, input, output):
    take_time_dict[layer_name] = time.time() - take_time_dict[layer_name]
