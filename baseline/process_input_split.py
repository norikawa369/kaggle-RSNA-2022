import gc
import glob
import os
import re

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydicom as dicom
import torch
import torchvision as tv
from sklearn.model_selection import GroupKFold
from torch.cuda.amp import GradScaler, autocast
from torchvision.models.feature_extraction import create_feature_extractor


# PATHs
CD = "/content"
RSNA_2022_PATH = f'{CD}/rsna-2022-cervical-spine-fracture-detection'
TRAIN_IMAGES_PATH = f'{RSNA_2022_PATH}/train_images'
TEST_IMAGES_PATH = f'{RSNA_2022_PATH}/test_images'

EFFNET_CHECKPOINTS_PATH = f'{CD}/rsna-2022-base-effnetv2'
METADATA_PATH = f'{CD}/vertebrae-detection-checkpoints'
MY_METADATA_PATH = f'{CD}/my-vertebrae-detection-checkpoints'

# config
N_FOLDS = 5



# load csv files
df_train = pd.read_csv(f'{RSNA_2022_PATH}/train.csv')

# rsna-2022-spine-fracture-detection-metadata contains inference of C1-C7 vertebrae for all training sample (95% accuracy)
df_train_slices = pd.read_csv(f'{METADATA_PATH}/train_segmented.csv')
c1c7 = [f'C{i}' for i in range(1, 8)]
df_train_slices[c1c7] = (df_train_slices[c1c7] > 0.5).astype(int)

# merge df_train and df_train_slices
df_train = df_train_slices.set_index('StudyInstanceUID').join(df_train.set_index('StudyInstanceUID'),
                                                            rsuffix='_fracture').reset_index().copy()
# drop the scan which does not include a full cervical spine
df_train = df_train.query('StudyInstanceUID != "1.2.826.0.1.3680043.20574"').reset_index(drop=True)

# group k fold
split = GroupKFold(N_FOLDS)
for k, (_, test_idx) in enumerate(split.split(df_train, groups=df_train.StudyInstanceUID)):
    df_train.loc[test_idx, 'split'] = k

# save csv file in content directory
df_train.to_pickle(f'{CD}/train_vert_det.pkl')

def load_dicom(path):
    """
    This supports loading both regular and compressed JPEG images. 
    See the first sell with `pip install` commands for the necessary dependencies
    """
    img = dicom.dcmread(path)
    img.PhotometricInterpretation = 'YBR_FULL'
    data = img.pixel_array
    data = data - np.min(data)
    if np.max(data) != 0:
        data = data / np.max(data)
    data = (data * 255).astype(np.uint8)
    return cv2.cvtColor(data, cv2.COLOR_GRAY2RGB), img

def img_preprocess(path):
    """
    image preprocessing
    脊椎が見えているところを拡大する
    """
    


