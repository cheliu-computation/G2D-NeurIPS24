# G2D-NeurIPS2024
G2D: From Global to Dense Radiography Representation Learning via Vision-Language Pre-training

[G2D: From Global to Dense Radiography Representation Learning via Vision-Language Pre-training](https://arxiv.org/abs/2312.01522), NeurIPS 2024.

###  Installation
To clone this repository:
```
git clone https://github.com/cheliu-computation/G2D-NeurIPS24.git
```
To install Python dependencies:
```
pip install -r requirements.txt
```
All experiments are implemented on A100 GPU.

### Pre-train Dataset downloading
Datasets we used are as follows:
- **MIMIC-CXR**: We downloaded the [MIMIC-CXR-JPG](https://physionet.org/content/mimic-cxr-jpg/2.0.0/) dataset as the radiographs. Paired medical reports can be downloaded in [MIMIC-CXR](https://physionet.org/content/mimic-cxr/2.0.0/mimic-cxr-reports.zip).

### Preprocessing
- First we follow [MGCA](https://github.com/HKU-MedAI/MGCA) preprocessing to extract a master csv includes all CXR scans associated with report. You can find in [Preprocessing](https://github.com/HKU-MedAI/MGCA/blob/main/mgca/preprocess/mimic_cxr.py). 
- Then, run 'ext_data.py' to extract all scans and save as a npy file. It will accelerate the pre-training stage.

### Pre-training
We pre-trained MGCA on MIMIC-CXR using this command:
```

cd /G2D-NeurIPS24/PRETRAIN
torchrun --nnodes=1 --nproc_per_node=8 main.py
```

### Finetune on downstream tasks
We evlauate the performance of G2D on three fine-tune downstream tasks: image classification, object detection, semantic segmentation and two zero-shot downstream tasks: zero-shot image classification, zero-shot image grounding.

For image classification, semantic segmentation and object detection, we follow [MGCA-NeurIPS2022](https://github.com/HKU-MedAI/MGCA) offical configuration and code. The dataset can be found in MGCA repository.

For zero-shot image classification and grounding tasks, we follow [MedKLIP-ICCV2023](https://github.com/MediaBrain-SJTU/MedKLIP), please follow their offical code to extract data and implement Image-Text Retrieval tasks.
