import gc
import glob
import os
import re
import pickle

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
from tqdm import tqdm

import wandb

# Effnet
WEIGHTS = tv.models.efficientnet.EfficientNet_V2_S_Weights.DEFAULT
RSNA_2022_PATH = '../input/rsna-2022-cervical-spine-fracture-detection'
TRAIN_IMAGES_PATH = f'{RSNA_2022_PATH}/train_images'
TEST_IMAGES_PATH = f'{RSNA_2022_PATH}/test_images'
EFFNET_MAX_TRAIN_BATCHES = 4000
EFFNET_MAX_EVAL_BATCHES = 200
ONE_CYCLE_MAX_LR = 0.0001
ONE_CYCLE_PCT_START = 0.3
SAVE_CHECKPOINT_EVERY_STEP = 1000
EFFNET_CHECKPOINTS_PATH = '../input/rsna-2022-base-effnetv2'
MY_EFFNET_CHECKPINTS_PATH = '../input/my-rsna2022-effnet-1'
FRAC_LOSS_WEIGHT = 2.
N_FOLDS = 5
METADATA_PATH = '../input/vertebrae-detection-checkpoints'
MY_METADATA_PATH = '../input/my-vertebrae-detection-checkpoints'
DICT_PATH = '../input/train-dict-and-list'

PREDICT_MAX_BATCHES = 1e9

os.environ['WANDB_API_KEY'] = 'yourkeyhere'

# Common
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
if DEVICE == 'cuda':
    BATCH_SIZE = 32
else:
    BATCH_SIZE = 2

# prepare dataframe
df_train = pd.read_csv(f'{RSNA_2022_PATH}/train.csv')

df_train_slices = pd.read_csv(f'{METADATA_PATH}/train_segmented.csv')
c1c7 = [f'C{i}' for i in range(1, 8)]
df_train_slices[c1c7] = (df_train_slices[c1c7] > 0.5).astype(int)

df_train = df_train_slices.set_index('StudyInstanceUID').join(df_train.set_index('StudyInstanceUID'),
                                                            rsuffix='_fracture').reset_index().copy()
df_train = df_train.query('StudyInstanceUID != "1.2.826.0.1.3680043.20574"').reset_index(drop=True)

# split
split = GroupKFold(N_FOLDS)
for k, (_, test_idx) in enumerate(split.split(df_train, groups=df_train.StudyInstanceUID)):
    df_train.loc[test_idx, 'split'] = k

# util fuctions
def window(img, WL=400, WW=1800):
    upper, lower = WL+WW//2, WL-WW//2
    X = np.clip(img.copy(), lower, upper)
    X = X - np.min(X)
    X = X / np.max(X)
    X = (X*255.0).astype('uint8')
    return X

def load_dicom(path):
    data = dicom.dcmread(path)
    x = data.pixel_array
    x = x*data.RescaleSlope+data.RescaleIntercept
    return x

def preprocess(path):
    # load dicom data
    img = load_dicom(path)
    # window processing
    img = window(img, WL=400, WW=1800)
    
    img = cv2.resize(img, dsize=(600, 600))
    
    z_pos = re.findall(r"\d+", path)[-1]
    z_pos = int(z_pos)
    
    if z_pos <= 30:
        img = img[30:430, 100:500]
    else:
        img = img[75:525, 75:525]
    
    img = cv2.resize(img, dsize=(512, 512))
    
    return img

# load dict
with open(f'{DICT_PATH}/series_dict.pkl', 'rb') as f:
    series_dict = pickle.load(f)
with open(f'{DICT_PATH}/image_dict.pkl', 'rb') as f:
    image_dict = pickle.load(f)


# Dataset
class EffnetDataSet(torch.utils.data.Dataset):
    def __init__(self, df, path, transforms=None):
        super().__init__()
        self.df = df
        self.path = path
        self.transforms = transforms

    def __getitem__(self, i):
        index_id2 = self.df.iloc[i].StudyInstanceUID + '.' +  str(self.df.iloc[i].Slice)
        index_id1 = image_dict[index_id2]["image_minus1"]
        slice_id1 = index_id1.split(".")[-1]
        index_id3 = image_dict[index_id2]["image_plus1"]
        slice_id3 = index_id3.split(".")[-1]
        path2 = os.path.join(self.path, self.df.iloc[i].StudyInstanceUID, f'{self.df.iloc[i].Slice}.dcm')
        path1 = os.path.join(self.path, self.df.iloc[i].StudyInstanceUID, f'{slice_id1}.dcm')
        path3 = os.path.join(self.path, self.df.iloc[i].StudyInstanceUID, f'{slice_id3}.dcm')
        
        try:
            img1 = preprocess(path1)
            img2 = preprocess(path2)
            img3 = preprocess(path3)
            img1 = np.expand_dims(img1, axis=2)
            img2 = np.expand_dims(img2, axis=2)
            img3 = np.expand_dims(img3, axis=2)
            
            x = np.concatenate([img1, img2, img3], axis=2)

            # Pytorch uses (batch, channel, height, width) order. Converting (height, width, channel) -> (channel, height, width)
            x = np.transpose(x, (2, 0, 1))
            
            if self.transforms is not None:
                x = self.transforms(torch.as_tensor(x))
        except Exception as ex:
            print(ex)
            return None

        if 'C1_fracture' in self.df:
            frac_targets = torch.as_tensor(self.df.iloc[i][['C1_fracture', 'C2_fracture', 'C3_fracture', 'C4_fracture',
                                                            'C5_fracture', 'C6_fracture', 'C7_fracture']].astype(
                'float32').values)
            vert_targets = torch.as_tensor(
                self.df.iloc[i][['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7']].astype('float32').values)
            frac_targets = frac_targets * vert_targets  # we only enable targets that are visible on the current slice
            return x, frac_targets, vert_targets
        return x

    def __len__(self):
        return len(self.df)

# Model
class EffnetModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        effnet = tv.models.efficientnet_v2_s(weights=WEIGHTS)
        self.model = create_feature_extractor(effnet, ['flatten'])
        self.nn_fracture = torch.nn.Sequential(
            torch.nn.Linear(1280, 7),
        )
        self.nn_vertebrae = torch.nn.Sequential(
            torch.nn.Linear(1280, 7),
        )

    def forward(self, x):
        # returns logits
        x = self.model(x)['flatten']
        return torch.sigmoid(self.nn_fracture(x)), torch.sigmoid(self.nn_vertebrae(x)), x

    def predict(self, x):
        frac, vert, x = self.forward(x)
        return torch.sigmoid(frac), torch.sigmoid(vert)


# loss fuction
def weighted_loss(y_pred_logit, y, reduction='mean', verbose=False):
    """
    Weighted loss
    We reuse torch.nn.functional.binary_cross_entropy_with_logits here. pos_weight and weights combined give us necessary coefficients described in https://www.kaggle.com/competitions/rsna-2022-cervical-spine-fracture-detection/discussion/340392

    See also this explanation: https://www.kaggle.com/code/samuelcortinhas/rsna-fracture-detection-in-depth-eda/notebook
    """

    neg_weights = (torch.tensor([7., 1, 1, 1, 1, 1, 1, 1]) if y_pred_logit.shape[-1] == 8 else torch.ones(y_pred_logit.shape[-1])).to(DEVICE)
    pos_weights = (torch.tensor([14., 2, 2, 2, 2, 2, 2, 2]) if y_pred_logit.shape[-1] == 8 else torch.ones(y_pred_logit.shape[-1]) * 2.).to(DEVICE)

    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        y_pred_logit,
        y,
        reduction='none',
    )

    if verbose:
        print('loss', loss)

    pos_weights = y * pos_weights.unsqueeze(0)
    neg_weights = (1 - y) * neg_weights.unsqueeze(0)
    all_weights = pos_weights + neg_weights

    if verbose:
        print('all weights', all_weights)

    loss *= all_weights
    if verbose:
        print('weighted loss', loss)

    norm = torch.sum(all_weights, dim=1).unsqueeze(1)
    if verbose:
        print('normalization factors', norm)

    loss /= norm
    if verbose:
        print('normalized loss', loss)

    loss = torch.sum(loss, dim=1)
    if verbose:
        print('summed up over patient_overall-C1-C7 loss', loss)

    if reduction == 'mean':
        return torch.mean(loss)
    return loss

# Train fuctions
def filter_nones(b):
    return torch.utils.data.default_collate([v for v in b if v is not None])

def save_model(name, model):
    torch.save(model.state_dict(), f'{name}.pth')

def load_model(model, name, path='.'):
    data = torch.load(os.path.join(path, f'{name}.pth'), map_location=DEVICE)
    model.load_state_dict(data)
    return model

def evaluate_effnet(model: EffnetModel, ds, max_batches=PREDICT_MAX_BATCHES, shuffle=False):
    torch.manual_seed(42)
    model = model.to(DEVICE)
    dl_test = torch.utils.data.DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle, num_workers=os.cpu_count(),
                                          collate_fn=filter_nones)
    pred_frac = []
    pred_vert = []
    with torch.no_grad():
        model.eval()
        frac_losses = []
        vert_losses = []
        with tqdm(dl_test, desc='Eval', miniters=10) as progress:
            for i, (X, y_frac, y_vert) in enumerate(progress):
                with autocast():
                    y_frac_pred, y_vert_pred = model.forward(X.to(DEVICE))
                    frac_loss = weighted_loss(y_frac_pred, y_frac.to(DEVICE)).item()
                    vert_loss = torch.nn.functional.binary_cross_entropy_with_logits(y_vert_pred, y_vert.to(DEVICE)).item()
                    pred_frac.append(torch.sigmoid(y_frac_pred))
                    pred_vert.append(torch.sigmoid(y_vert_pred))
                    frac_losses.append(frac_loss)
                    vert_losses.append(vert_loss)

                if i >= max_batches:
                    break
        return np.mean(frac_losses), np.mean(vert_losses), torch.concat(pred_frac).cpu().numpy(), torch.concat(pred_vert).cpu().numpy()

def gc_collect():
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == '__main__':
    model = load_model(EffnetModel(), f'effnetv2_1-all-8000', MY_EFFNET_CHECKPINTS_PATH)

    batch_size = 480

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    model.to(DEVICE)

    ds_train = EffnetDataSet(df_train, TRAIN_IMAGES_PATH, WEIGHTS.transforms())
    dl_train = torch.utils.data.DataLoader(ds_train, batch_size=batch_size, shuffle=False, num_workers=os.cpu_count(),
                                            collate_fn=filter_nones, pin_memory=True)


    feature = np.zeros((len(ds_train), 1280),dtype=np.float32)
    frac_pred_prob = np.zeros((len(ds_train), 7),dtype=np.float32)
    vert_pred_prob = np.zeros((len(ds_train), 7),dtype=np.float32)

    model.eval()
        
    with tqdm(dl_train, desc='Feature', miniters=10) as progress:
        
        start = 0
        for i, (images, y_frac, y_vert) in enumerate(progress):
            with torch.no_grad():
                end = start+len(images)
                with autocast():
                    y_frac_pred, y_vert_pred, features = model(images.to(DEVICE))
                feature[start:end] = np.squeeze(features.cpu().data.numpy())
                frac_pred_prob[start:end] = np.squeeze(y_frac_pred.cpu().data.numpy())
                vert_pred_prob[start:end] = np.squeeze(y_vert_pred.cpu().data.numpy())
            start = end
    

    np.save('feature_train', feature)
    np.save('frac_pred_prob_train', frac_pred_prob)
    np.save('vert_pred_prob_train', vert_pred_prob)
