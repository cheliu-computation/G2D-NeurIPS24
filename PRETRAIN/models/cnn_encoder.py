import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision import models as models_2d
from torchvision.models import ResNet18_Weights, ResNet34_Weights, ResNet50_Weights, ResNet101_Weights


class Identity(nn.Module):
    """Identity layer to replace last fully connected layer"""

    def forward(self, x):
        return x


################################################################################
# ResNet Family
################################################################################

PRETRAIN = True

def resnet_18(pretrained=PRETRAIN):
    if pretrained:
        model = models_2d.resnet18(weights=ResNet18_Weights.DEFAULT)
    else:
        model = models_2d.resnet18(weights=None)
    final_channels = model.fc.in_features
    model.fc = Identity()
    return model, final_channels


def resnet_34(pretrained=PRETRAIN):
    if pretrained:
        model = models_2d.resnet34(weights=ResNet34_Weights.DEFAULT)
    else:
        model = models_2d.resnet34(weights=None)
    final_channels = model.fc.in_features
    model.fc = Identity()
    return model, final_channels


def resnet_50(pretrained=PRETRAIN):
    if pretrained:
        model = models_2d.resnet50(weights=ResNet50_Weights.DEFAULT)
    else:
        model = models_2d.resnet50(weights=None)
    final_channels = model.fc.in_features
    model.fc = Identity()
    return model, final_channels

def resnet_101(pretrained=PRETRAIN):
    if pretrained:
        model = models_2d.resnet101(weights=ResNet101_Weights.DEFAULT)
    else:
        model = models_2d.resnet101(weights=None)
    final_channels = model.fc.in_features
    model.fc = Identity()
    return model, final_channels

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
        x, att_map = self.mhsa(x[:1], x, x, average_attn_weights=True)
        x = self.c_proj(x)
        # original CLIP return att_map as the visual grounding map
        # MaskCLIP+ return x before MHSA as the visual grounding map
        return x.squeeze(0), att_map.squeeze(0)
    
class cnn_backbone(nn.Module):
    def __init__(self, backbone, heads=4, input_resolution=224, width=128):
        super(cnn_backbone, self).__init__()
        assert backbone in ['resnet18', 'resnet34', 'resnet50', 'resnet101'] # only support resnet18, resnet34, resnet50, resnet101
        if backbone == 'resnet18':
            self.backbone, self.final_channels = resnet_18()
        elif backbone == 'resnet34':
            self.backbone, self.final_channels = resnet_34()
        elif backbone == 'resnet50':
            self.backbone, self.final_channels = resnet_50()
        elif backbone == 'resnet101':
            self.backbone, self.final_channels = resnet_101()

        embed_dim = self.final_channels  # the ResNet feature dimension
        output_dim = embed_dim // 1
        # self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        def stem(self, x):
            x = self.backbone.conv1(x)
            x = self.backbone.bn1(x)
            x = self.backbone.relu(x)
            x = self.backbone.maxpool(x)
            return x
        
        x = stem(self, x)           
        x = self.backbone.layer1(x) # 64, 56, 56
        x = self.backbone.layer2(x) # 128, 28, 28
        x = self.backbone.layer3(x) # 256, 14, 14
        x = self.backbone.layer4(x) # 512, 7, 7

        if self.training:
            # x, _ = self.attnpool(x)
            # x = self.avgpool(x).flatten(1)
            return x
        else:
            # x, att_map = self.attnpool(x)
            # x = self.avgpool(x).flatten(1)
            return x, None