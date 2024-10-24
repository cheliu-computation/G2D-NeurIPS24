from cgi import test
import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModel, AutoTokenizer
from cnn_encoder import cnn_backbone

class Pendingmodel(torch.nn.Module):
    def __init__(self, network_config, device_id):
        super(Pendingmodel, self).__init__()
        self.device_id=device_id

        self.proj_hidden = network_config['projection_head']['mlp_hidden_size']
        self.proj_out = network_config['projection_head']['projection_size']

        # define image encoder
        self.encoder = cnn_backbone(backbone='resnet50', input_resolution=network_config['img_size'])

        # define text encoder
        url = network_config['text_model']
        self.lm_model = AutoModel.from_pretrained(
            url, trust_remote_code=True, revision='main')
        self.tokenizer = AutoTokenizer.from_pretrained(
            url, trust_remote_code=True, revision='main')
        
        self.proj_v = nn.Sequential(
            nn.Conv2d(2048, self.proj_hidden, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.proj_hidden, affine=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.proj_hidden, self.proj_out, kernel_size=1, stride=1, padding=0, bias=False),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # linear projection head
        # conv projection head
        self.proj_t = nn.Sequential(
            nn.Conv1d(768, self.proj_hidden, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm1d(self.proj_hidden, affine=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.proj_hidden, self.proj_out, kernel_size=1, stride=1, padding=0, bias=False),
            # nn.BatchNorm2d(self.proj_out, affine=False)
        )
        
        # image local attention layer
        self.img_local_atten_layer = nn.MultiheadAttention(
            self.proj_out, 1, batch_first=True)
        # token local attention layer
        self.token_local_atten_layer = nn.MultiheadAttention(
            self.proj_out, 1, batch_first=True)

    def _tokenize(self, text):
        if self.training:
            tokenizer_output = self.tokenizer.batch_encode_plus(batch_text_or_text_pairs=text,
                                                                add_special_tokens=True,
                                                                truncation=True,
                                                                max_length=256,
                                                                padding='max_length',
                                                                return_tensors='pt')
        else:
            tokenizer_output = self.tokenizer.batch_encode_plus(batch_text_or_text_pairs=text,
                                                                add_special_tokens=True,
                                                                padding='longest',
                                                                return_tensors='pt')

        return tokenizer_output.input_ids, tokenizer_output.attention_mask

    # get text embedding
    # @torch.no_grad()
    def get_text_emb(self, text):
        input_ids, attention_mask = self._tokenize(text)
        text_emb = self.lm_model(input_ids=input_ids.to(self.device_id),
                                 attention_mask=attention_mask.to(self.device_id)).last_hidden_state
        return text_emb
    
    # get image embedding
    def get_img_emb(self, img):
        img_emb = self.encoder(img)
        return img_emb

    # get image embedding and grounding map
    @torch.no_grad()
    def get_img_emb_map(self, img):
        assert self.encoder.training is False
        img_emb, _ = self.encoder(img)
        return img_emb, None
    
    # get prompt embedding for zeroshot cls
    @torch.no_grad()
    def get_prompt_emb(self, prompt):
        prompt_emb = self.get_text_emb(prompt)
        prompt_emb = self.proj_t(prompt_emb.permute(0, 2, 1).contiguous()) # [b, 128, num_tokens+1]
        prompt_emb = prompt_emb.permute(0, 2, 1).contiguous()
        return prompt_emb[:, 0]
    
    # get image proj embedding
    @torch.no_grad()
    def get_img_proj_emb(self, img):
        img_emb, _ = self.get_img_emb_map(img)
        proj_img_emb = self.proj_v(img_emb)
        proj_img_emb = self.avgpool(proj_img_emb).squeeze()
        return proj_img_emb
    
    def forward(self, img, text):
        img_emb = self.get_img_emb(img)

        # pooler_output: [b, 1, 768]
        text_emb = self.get_text_emb(text) # [b, num_tokens+1, 768]

        # project to _ dim
        proj_img_emb = self.proj_v(img_emb)
        proj_img_emb_pool = self.avgpool(proj_img_emb).squeeze()

        # proj_text_emb = self.proj_t(text_emb[:, 0].contiguous())
        proj_text_emb = self.proj_t(text_emb.permute(0, 2, 1).contiguous()) # [b, 128, num_tokens+1]
        proj_text_emb = proj_text_emb.permute(0, 2, 1).contiguous()

        # local attention
        proj_img_emb = proj_img_emb.reshape(proj_img_emb.shape[0], self.proj_out, -1).permute(0, 2, 1).contiguous()
        proj_img_emb = F.normalize(proj_img_emb, dim=-1)
        proj_text_emb = F.normalize(proj_text_emb, dim=-1)
        
        return {'img_emb': img_emb,
                'proj_img_emb': proj_img_emb_pool,
                'patch_emb': None,
                'proj_text_emb': proj_text_emb[:, 0],
                'word_emb': None,
                }

