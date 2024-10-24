import random
import tempfile
import os
import pandas as pd
import numpy as np
import yaml
import sys
sys.path.append('G2D-NeurIPS24/PRETRAIN/utils')
sys.path.append('G2D-NeurIPS24/models')
from utils.trainer import Trainer
from utils.dataset import VLP_dataset

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import torch.distributed as dist

from models.pre_model import Pendingmodel
from models.cnn_ae import CNN_AE


os.environ["TOKENIZERS_PARALLELISM"] = "true"


def ddp_main():
    dist.init_process_group("nccl")
    torch.cuda.empty_cache()
    rank = dist.get_rank()

    print(f"Start running basic DDP example on rank {rank}.")
    device_id = rank % torch.cuda.device_count()

    # set up
    config = yaml.load(open("config.yaml", "r"), Loader=yaml.FullLoader)

    torch.manual_seed(42)
    random.seed(0)
    np.random.seed(0)

    # loading data path
    text_path = config['text_path']
    img_path = config['img_path']

    # define image-text dataset
    train_dataset = VLP_dataset(
        image_path=img_path, csv_path=text_path)
    train_dataset = train_dataset.get_dataset(train_test='train')

    # building model part
    # --------------------
    if config['network']['decoder'] == 'no':
        model = Pendingmodel(config['network'], device_id=device_id)
    elif config['network']['decoder'] == 'yes':
        model = CNN_AE(config['network'], device_id=device_id)

    '''
    you can freeze bert from last layer to first layer.
    set num of layer in config.yaml
    default is freeze 9 layers
    '''
    if config['network']['free_layers'] is not None:
        for layer_idx in range(int(config['network']['free_layers'])):
            for param in list(model.lm_model.encoder.layer[layer_idx].parameters()):
                param.requires_grad = False

    model = model.to(device_id)
    model = DDP(model, device_ids=[device_id], find_unused_parameters=True)

    # --------------------

    # choose optimizer (no LARS, AdamW with small batch)
    # --------------------
    optimizer = torch.optim.AdamW(
        model.parameters(),
        **config['optimizer']['params'],
        betas=(0.9, 0.999)
    )

    # ---------xw-----------
    trainer = Trainer(model=model,
                            optimizer=optimizer,
                            device=device_id,
                            model_name=config['wandb_name'],
                            **config['trainer'])
    # --------------------
    
    # --------------------
    # I_T_P_trainer
    trainer.train_process(train_dataset)


ddp_main()
