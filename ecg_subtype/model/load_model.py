import os
from symtable import Class

import torch
import yaml

import logging

from torch import nn

from model import st_mem_vit

logger = logging.getLogger(__name__)



def load_ecg_model(config_path, freeze_backbone=0):
    with open(os.path.realpath(config_path), 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    # config['model']['num_classes'] = 2
    model_name = config['model_name']
    if model_name in st_mem_vit.__dict__:
        ecg_model = st_mem_vit.__dict__[model_name](**config['model'])
    else:
        raise ValueError(f'Unsupported model name: {model_name}')

    if config['mode'] != "scratch":
        checkpoint = torch.load(config['encoder_path'], map_location='cpu')
        print(f"Load pre-trained checkpoint from: {config['encoder_path']}")
        checkpoint_model = checkpoint['model']
        state_dict = ecg_model.state_dict()
        # do not load head.weight and head.bias for classification in the original pre-trained model
        # for k in ['head.weight', 'head.bias']:
        #     if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
        #         print(f"Remove key {k} from pre-trained checkpoint")
        #         del checkpoint_model[k]
        msg = ecg_model.load_state_dict(checkpoint_model, strict=False)
        print(f'Load pre-trained ECG model: {msg}')
        # assert set(msg.missing_keys) == {'head.weight', 'head.bias'}
        if freeze_backbone == 1:
            ecg_model.freeze_backbone()
            print("Backbone model is frozen.")
    return ecg_model
