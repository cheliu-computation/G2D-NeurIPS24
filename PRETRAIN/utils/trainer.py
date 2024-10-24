# package import
import os
from typing import Type
from PIL import Image
from matplotlib import pyplot as plt
import torch
import torch.nn.functional as F
import torchvision as tv
import pandas as pd
from torch.utils.data.dataloader import DataLoader
# import wandb
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler as GradScaler
from tqdm import tqdm
import numpy as np
import torch.nn as nn
from pprint import pprint

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler

from loss import clip_loss, bceDiceLoss


class Trainer:
    def __init__(self, model,
                 optimizer, device, model_name, **args):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.model_name = model_name
        self.train_batch_size = args['batch_size']
        self.max_epochs = args['max_epochs']
        self.lr_max = args['lr']
        self.num_workers = args['num_workers']
        self.decoder = args['decoder']
        self.crop_img_size = args['crop_img_size']
        self.merge_threshold = args['merge_threshold']
        self.num_pseudo_map = args['num_pseudo_map']
        self.quantil = args['quantil']
        self.epoch_num = 0
        self.global_step = 0
        self.resize = tv.transforms.Resize((self.crop_img_size, self.crop_img_size), interpolation=Image.NEAREST)
        print(f'trainer init successful!, quantile is {self.quantil}')
        print('---------------------------')
        
    def create_scheduler(self):
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=200, eta_min=1e-8)


    def train_epoch(self, train_loader, scheduler, scaler):
        epoch_loss = 0
        epoch_acc1 = []
        epoch_acc5 = []
        epoch_global, epoch_patch, epoch_word, epoch_seg, epoch_reconst = [], [], [], [], []

        for data in tqdm(train_loader):
            text, img, raw_img = self.prepare_data(data)
            
            loss, acc1, acc5, metric = self.train_batch(text, img, raw_img, scaler, scheduler)
            self.global_step += 1

            epoch_loss += loss.item()
            epoch_acc1.append(acc1.item())
            epoch_acc5.append(acc5.item())
            epoch_global.append(metric['clip_loss'])
            epoch_patch.append(metric['patch_loss'])
            epoch_word.append(metric['word_loss'])
            epoch_seg.append(metric['seg_loss'])
            epoch_reconst.append(metric['reconst_loss'])

        epoch_acc1 = np.array(epoch_acc1).mean()
        epoch_acc5 = np.array(epoch_acc5).mean()
        epoch_global = np.array(epoch_global).mean()
        epoch_patch = np.array(epoch_patch).mean()
        epoch_word = np.array(epoch_word).mean()
        epoch_seg = np.array(epoch_seg).mean()
        epoch_reconst = np.array(epoch_reconst).mean()

        metric = {'global_loss': epoch_global,
                    'patch_loss': epoch_patch,
                    'word_loss': epoch_word,
                    'seg_loss': epoch_seg,
                    'reconst_loss': epoch_reconst
                    }
        return epoch_loss, epoch_acc1, epoch_acc5, metric


    def prepare_data(self, data):
        text = data['raw_text']
        img = data['image'].to(torch.float32).to(self.device).contiguous()
        raw_img = np.array(data['raw_image'])
        return text, img, raw_img

    def transform_att2label(self, att_map):
        pseudo_map = att_map.detach().cpu() 

        if self.num_pseudo_map == 1:

            pseudo_map = torch.sum(pseudo_map, dim=1) / 4 

            pseudo_map = self.resize(pseudo_map).numpy()

            if self.merge_threshold == 'mean':
                threshold = np.mean(pseudo_map, axis=(1, 2), keepdims=True)
            elif self.merge_threshold == 'median':
                threshold = np.median(pseudo_map, axis=(1, 2), keepdims=True)
            elif self.merge_threshold == 'quantil':
                threshold = np.quantile(pseudo_map, self.quantil, axis=(1, 2), keepdims=True)

            pseudo_map[pseudo_map > threshold] = 1
            pseudo_map[pseudo_map <= threshold] = 0
            pseudo_map = pseudo_map.reshape(pseudo_map.shape[0], 1, pseudo_map.shape[1], pseudo_map.shape[2])

        # multi pseudo dense label
        else:
            # resize to each head map original image size
            pseudo_map = self.resize(pseudo_map).numpy()
            
        # merge threshold
            if self.merge_threshold == 'mean':
                threshold = np.mean(pseudo_map, axis=(2, 3), keepdims=True)
            elif self.merge_threshold == 'median':
                threshold = np.median(pseudo_map, axis=(2, 3), keepdims=True)
            elif self.merge_threshold == 'quantil':
                threshold = np.quantile(pseudo_map, self.quantil, axis=(2, 3), keepdims=True)

            pseudo_map[pseudo_map > threshold] = 1
            pseudo_map[pseudo_map <= threshold] = 0
        
        return pseudo_map

# modfiy here for wo encoder and w encoder training
    def train_batch(self, text, img, raw_img, scaler, scheduler):
        total_local_loss = None
        loss_patch = None
        loss_word = None
        loss_seg = None
        loss_reconst = None

        self.optimizer.zero_grad()
        with autocast():
            output_dict = self.model(img, text)
            if self.decoder == 'no':
                _, proj_img_emb, proj_text_emb = output_dict['img_emb'], output_dict['proj_img_emb'], output_dict['proj_text_emb']

                global_loss, acc1, acc5 = clip_loss(proj_img_emb, proj_text_emb, device=self.device)
                loss = global_loss

            elif self.decoder == 'yes':
                _, pool_proj_img_emb, proj_text_emb = output_dict['img_emb'], output_dict['pool_proj_img_emb'], output_dict['proj_text_emb']
                global_loss, acc1, acc5 = clip_loss(pool_proj_img_emb, proj_text_emb, device=self.device)

                # seg loss part
                p_dense_map = output_dict['att_map']
                p_dense_map = self.transform_att2label(p_dense_map)
                p_dense_map = torch.tensor(p_dense_map).to(self.device)
                seg_img = output_dict['img_dec']
                loss_seg = bceDiceLoss(seg_img, p_dense_map)

                p_dense_map = p_dense_map.detach().cpu()
                
                if self.device == 0:
                    print('current gpu memory:', torch.cuda.memory_allocated(self.device)/1024/1024/1024)
                # we warm up the model with global loss
                if self.global_step < 200:
                    loss = global_loss
                else:
                    loss = global_loss + loss_seg
                    
                
            scaler.scale(loss).backward()

            scaler.step(self.optimizer)
            scaler.update()
            scheduler.step()
            
            if loss_patch is None:
                loss_patch = torch.tensor(0)
            if loss_patch is None:
                loss_patch = torch.tensor(0)
            if loss_word is None:
                loss_word = torch.tensor(0)
            if loss_seg is None:
                loss_seg = torch.tensor(0)
            if loss_reconst is None:
                loss_reconst = torch.tensor(0)

            metric = {'clip_loss': global_loss.item(),
                      'patch_loss': loss_patch,
                      'word_loss': loss_word,
                      'seg_loss': loss_seg.item(),
                      'reconst_loss': loss_reconst.item()}

        return loss, acc1, acc5, metric
    
    # traing process
    def train_process(self, train_dataset):

        train_loader = self.create_data_loader(train_dataset)
        model_checkpoints_folder = self.prepare_checkpoint_directory()
        start_epoch, is_continued = self.load_checkpoint_if_exists(model_checkpoints_folder)

        self.epoch_num = start_epoch
        self.global_step = start_epoch * (len(train_dataset)//self.train_batch_size//8)

        total_metric = {'global_loss': [],
                        'patch_loss': [],
                        'word_loss': [],
                        'seg_loss': [],
                        'reconst_loss': [],
                        'acc1': [],
                        'acc5': []
                        }
        
        print('training start!')
        scheduler = self.create_scheduler()
        scaler = GradScaler()

        for epoch_counter in tqdm(range(start_epoch, self.max_epochs+1)):
            epoch_loss, epoch_acc1, epoch_acc5, metric = self.train_epoch(train_loader, scheduler, scaler)
            self.log_and_save_model(train_dataset, epoch_counter, epoch_loss, epoch_acc1, epoch_acc5, metric, model_checkpoints_folder)

            total_metric['global_loss'].append(metric['global_loss'])
            total_metric['patch_loss'].append(metric['patch_loss'])
            total_metric['word_loss'].append(metric['word_loss'])
            total_metric['seg_loss'].append(metric['seg_loss'])
            total_metric['reconst_loss'].append(metric['reconst_loss'])
            total_metric['acc1'].append(epoch_acc1)
            total_metric['acc5'].append(epoch_acc5)

            self.epoch_num += 1
        self.save_final_model(model_checkpoints_folder, total_metric)


    def create_data_loader(self, train_dataset):
        return DataLoader(train_dataset, batch_size=self.train_batch_size, num_workers=self.num_workers,
                        drop_last=True, shuffle=False, sampler=DistributedSampler(train_dataset), pin_memory=True)


    def prepare_checkpoint_directory(self):
        
        model_checkpoints_folder = os.path.join(f'../checkpoints/{self.model_name}/')
        if not os.path.exists(model_checkpoints_folder):
            print('create directory "{}" for save checkpoint!'.format(model_checkpoints_folder))
            print('---------------------------')
            os.makedirs(model_checkpoints_folder)
        else:
            print('directory "{}" existing for save checkpoint!'.format(model_checkpoints_folder))
        return model_checkpoints_folder


    def load_checkpoint_if_exists(self, model_checkpoints_folder):
        print('#########################################')
        print('Be patient..., checking checkpoint now...')
        if os.path.exists(model_checkpoints_folder + self.model_name+'_6_checkpoint.pth'):
            ckpt = torch.load(model_checkpoints_folder + self.model_name+'_6_checkpoint.pth',
                            map_location='cpu')
            start_epoch = ckpt['epoch']
            # add module before all state keys
            new_state_dict = {}
            for k, v in ckpt['model_state_dict'].items():
                name = 'module.' + k
                new_state_dict[name] = v
            ckpt['model_state_dict'] = new_state_dict

            self.model.load_state_dict(ckpt['model_state_dict'])
            self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            print('continue training successful!')
            return start_epoch, True
        else:
            print('Start training from 0 epoch')
            return 0, False


    def log_and_save_model(self, train_dataset, epoch_counter, epoch_loss, epoch_acc1, epoch_acc5, metric, model_checkpoints_folder):
        if self.device == 0:
            epoch_iter = (len(train_dataset)//self.train_batch_size)
            print(f'{self.epoch_num} epoch loss is {epoch_loss/epoch_iter}, acc1 is {epoch_acc1}, acc5 is {epoch_acc5}')
            pprint(metric)
            if self.epoch_num % 2 == 0:
                self.save_checkpoints(self.epoch_num, model_checkpoints_folder)


    def save_checkpoints(self, epoch, model_checkpoints_folder):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.module.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()},
            model_checkpoints_folder + self.model_name+f'_{epoch}'+'_checkpoint.pth')
        if self.decoder == 'no':
            torch.save(self.model.module.encoder.state_dict(),
                    model_checkpoints_folder + self.model_name+f'_{epoch}'+'_encoder.pth')
        else:
            torch.save(self.model.module.img_model.encoder.state_dict(),
                    model_checkpoints_folder + self.model_name+f'_{epoch}'+'_encoder.pth')
            torch.save(self.model.module.img_model.state_dict(),
                    model_checkpoints_folder + self.model_name+f'_{epoch}'+'_encoder_decoder.pth')
            
    def save_final_model(self, model_checkpoints_folder, total_metric):
        if self.decoder == 'no':
            torch.save(self.model.module.encoder.state_dict(),
                    model_checkpoints_folder + self.model_name+'_encoder.pth')
        else:
            torch.save(self.model.module.img_model.encoder.state_dict(),
                    model_checkpoints_folder + self.model_name +'_encoder.pth')
            torch.save(self.model.module.img_model.state_dict(),
                    model_checkpoints_folder + self.model_name +'_encoder_decoder.pth')
            
        # torch.save(self.model.module.state_dict(),
        #         model_checkpoints_folder + self.model_name+'_total.pth')

        print('training finished!')