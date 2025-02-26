import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModel, AutoTokenizer
import segmentation_models_pytorch as smp

class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)

        self.mhsa = nn.MultiheadAttention(embed_dim, num_heads)        
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.flatten(start_dim=2).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, att_map = self.mhsa(x[:1], x, x, average_attn_weights=False)
        x = self.c_proj(x)
        # original CLIP return att_map as the visual grouning map
        # MaskCLIP+ return x before MHSA as the visual grouning map
        # return x.squeeze(0), att_map.squeeze(0)
        return x.squeeze(0), att_map.squeeze(2)
    
class CNN_AE(torch.nn.Module):
    def __init__(self, network_config, device_id):
        super(CNN_AE, self).__init__()
        self.device_id=device_id

        self.proj_hidden = network_config['projection_head']['mlp_hidden_size']
        self.proj_out = network_config['projection_head']['projection_size']

        # define image Unet backbone 
        self.img_model = smp.Unet(network_config['img_model'],
                                 in_channels=3,
                                   classes=network_config['unet_out_channel'],
                                     encoder_weights='imagenet')
        
        # define attention pooling
        # self.attnpool = AttentionPool2d(network_config['img_size'] // 32,
        #                                  self.img_model.encoder.out_channels[-1],
        #                                    network_config['att_pool_head'],
        #                                      self.img_model.encoder.out_channels[-1])
        
        # input 128, output 128
        self.attnpool = AttentionPool2d(spacial_dim = network_config['img_size'] // 32, 
                                        embed_dim = self.proj_out, 
                                        num_heads = network_config['att_pool_head'], 
                                        output_dim = self.proj_out)
        
        # define text encoder
        url = network_config['text_model']
        self.lm_model = AutoModel.from_pretrained(
            url, trust_remote_code=True, revision='main')
        self.tokenizer = AutoTokenizer.from_pretrained(
            url, trust_remote_code=True, revision='main')
        
        # define vision projection head
        # self.proj_v = nn.Sequential(
        #     nn.Linear(self.img_model.encoder.out_channels[-1], self.proj_hidden),
        #     nn.LayerNorm(self.proj_hidden),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(self.proj_hidden, self.proj_out),
        #     nn.LayerNorm(self.proj_out, elementwise_affine=False))
        
        self.proj_v = nn.Sequential(
            nn.Conv2d(self.img_model.encoder.out_channels[-1], 
                      self.proj_hidden, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.proj_hidden, affine=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.proj_hidden, 
                      self.proj_out, kernel_size=1, stride=1, padding=0, bias=False),
        )

        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))


        # define text projection head
        # self.proj_t = nn.Sequential(
        #     nn.Linear(768, self.proj_hidden),
        #     nn.LayerNorm(self.proj_hidden),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(self.proj_hidden, self.proj_out),
            # nn.LayerNorm(self.proj_out, elementwise_affine=False))

        self.proj_t = nn.Sequential(
            nn.Conv1d(768, self.proj_hidden, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm1d(self.proj_hidden, affine=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.proj_hidden, self.proj_out, kernel_size=1, stride=1, padding=0, bias=False),
        )

        # define 1x1 conv for pseudo map
        self.pseudo_conv = nn.Conv2d(self.img_model.encoder.out_channels[-2], self.proj_out, 1, 1, 0)

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
    def get_text_emb(self, text):
        input_ids, attention_mask = self._tokenize(text)
        text_emb = self.lm_model(input_ids=input_ids.to(self.device_id),
                                 attention_mask=attention_mask.to(self.device_id)).last_hidden_state
        return text_emb
    
    # get image embedding
    def get_img_emb(self, img):
        if self.training:
            pyramid_fea = self.img_model.encoder(img)
            # img_emb, att_map = self.attnpool(pyramid_fea[-1])
            img_emb = pyramid_fea[-1]
            proj_img_emb = self.proj_v(img_emb)
            pool_proj_img_emb, att_map = self.attnpool(proj_img_emb)
        else:
            with torch.no_grad():
                pyramid_fea = self.img_model.encoder(img)
                # img_emb, att_map = self.attnpool(pyramid_fea[-1])
                img_emb = pyramid_fea[-1]
                proj_img_emb = self.proj_v(img_emb)
                pool_proj_img_emb, att_map = self.attnpool(proj_img_emb)
        return {'pyramid_fea': pyramid_fea,
                'img_emb': img_emb,
                'proj_img_emb': proj_img_emb,
                'pool_proj_img_emb': pool_proj_img_emb,
                'att_map': att_map}

    # get image decoder output
    def get_img_dec(self, *pyarmid_fea):
        if self.training:
            img_dec = self.img_model.decoder(*pyarmid_fea)
            img_dec = self.img_model.segmentation_head(img_dec)
        else:
            with torch.no_grad():
                img_dec = self.img_model.decoder(*pyarmid_fea)
                img_dec = self.img_model.segmentation_head(img_dec)
        return img_dec

    # get prompt embedding for zeroshot cls
    @torch.no_grad()
    def get_prompt_emb(self, prompt):
        prompt_emb = self.get_text_emb(prompt)
        # prompt_emb = self.proj_t(prompt_emb[:, 0].contiguous())
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
        multi_fea = self.get_img_emb(img)
        pyramid_fea = multi_fea['pyramid_fea']
        img_emb = multi_fea['img_emb']
        proj_img_emb = multi_fea['proj_img_emb']
        pool_proj_img_emb = multi_fea['pool_proj_img_emb']
        att_map = multi_fea['att_map']

        img_dec = self.get_img_dec(*pyramid_fea)

        # pooler_output: [b, 1, 768]
        text_emb = self.get_text_emb(text)

        # visual project
        # proj_img_emb = self.proj_v(img_emb)
        proj_img_emb = proj_img_emb.reshape(proj_img_emb.shape[0], self.proj_out, -1).permute(0, 2, 1).contiguous()
        proj_img_emb = F.normalize(proj_img_emb, dim=-1)

        # text project
        # proj_text_emb = self.proj_t(text_emb[:, 0].contiguous())
        proj_text_emb = self.proj_t(text_emb.permute(0, 2, 1).contiguous()) # [b, 128, num_tokens+1]
        proj_text_emb = proj_text_emb.permute(0, 2, 1).contiguous()
        proj_text_emb = F.normalize(proj_text_emb, dim=-1)

        return {'img_emb': img_emb,
                'img_dec': img_dec,
                'patch_emb': proj_img_emb,
                'pool_proj_img_emb': pool_proj_img_emb,
                'proj_text_emb': proj_text_emb[:, 0],
                'word_emb': proj_text_emb[:, 1:],
                'att_map': att_map[:, :, 1:].reshape(img.shape[0], 4, img.shape[2]//32, img.shape[3]//32)}
