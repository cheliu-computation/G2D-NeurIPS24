import torch
import pandas as pd
from torch.utils.data import Dataset
import numpy as np
from torchvision.transforms import transforms
from PIL import Image


class IaT_embed_dataset(Dataset):
    def __init__(self, image_data, transform=None, **args):
        self.img_data = image_data

        self.text_csv = args['text']
        self.mode = args['train_test']
        self.transform = transform

    def __len__(self):
        return (self.img_data.shape[0])

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # get image
        image = self.img_data[idx]
        image = Image.fromarray(image).convert("RGB")

        # get raw text
        findings = self.text_csv['findings'].iloc[idx]
        impression = self.text_csv['impression'].iloc[idx]
        if findings == 'dumb' or type(findings) == float:
            pass
        else:
            impression += findings
            impression = impression.replace('IMPRESSSION:', '')
            impression = impression.replace('IMPRESSSION', '')
            impression = impression.replace('FINDINGS:', '')
            impression = impression.replace('FINDINGS', '')
            impression = impression.replace('COMPARISON:  ___', '')
            impression = impression.replace('COMPARISON:', '')
            impression = impression.replace('COMPARISON', '')
            
            m_impression = impression.split('.')
            if len(m_impression) > 1:
                m_impression = [a.strip() for a in m_impression]
                m_impression = [a for a in m_impression if not a.isdigit()]
                impression = '. '.join(m_impression)
            else:
                pass
    
        
        text = impression

        sample = {'raw_image': image, 'raw_text': text}

        if self.transform:
            # for 2 branch contrastive vision model (not useful for CLIP)
            if self.mode == 'train':
                sample['image'] = self.transform(sample['raw_image'])
                sample['raw_image'] = np.array(sample['raw_image'])
            else:
                sample['image'] = self.transform(sample['raw_image'])
                sample['raw_image'] = np.array(sample['raw_image'])
        return sample


class VLP_dataset:

    def __init__(self, image_path, csv_path):
        self.image_path = image_path
        self.csv_path = csv_path

    def get_dataset(self, train_test, T=None):
        normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

        if train_test == 'train':
            print('Apply Train-stage Transform!')

            Transforms = transforms.Compose([
                transforms.CenterCrop(256),
                transforms.RandomAffine(degrees=15, shear=15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                normalize
            ])
        else:
            print('Apply Test-stage Transform!')

            Transforms = transforms.Compose([
                transforms.CenterCrop(256),
                transforms.ToTensor(),
                normalize
            ])

        img_path = np.load(
            self.image_path, allow_pickle=True, mmap_mode='r')
        csv_path = pd.read_csv(
            self.csv_path, low_memory=False)

        misc_args = {'train_test': train_test,
                   'text': csv_path}

        dataset = IaT_embed_dataset(image_data=img_path,
                                       transform=Transforms,
                                       **misc_args)

        return dataset
