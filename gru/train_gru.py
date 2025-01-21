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
import torch.nn as nn
from torch import optim
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
GRU_CHECKPOINTS_PATH = '../input/gru-checkpoints'
FRAC_LOSS_WEIGHT = 2.
N_FOLDS = 5
METADATA_PATH = '../input/vertebrae-detection-checkpoints'
MY_METADATA_PATH = '../input/my-vertebrae-detection-checkpoints'
DICT_PATH = '../input/train-dict-and-list'
FEATURE_PATH = '../output'


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

# utils
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# load dict
with open(f'{DICT_PATH}/series_dict.pkl', 'rb') as f:
    series_dict = pickle.load(f)
with open(f'{DICT_PATH}/image_dict.pkl', 'rb') as f:
    image_dict = pickle.load(f)
    
feature_train = np.load(f'{FEATURE_PATH}/feature_train_all_8000.npy')
feature_train1 = np.load(f'{FEATURE_PATH}/frac_pred_prob_train_all_8000.npy')
feature_train2 = np.load(f'{FEATURE_PATH}/vert_pred_prob_train_all_8000.npy')

df_train["image_id"] = df_train["StudyInstanceUID"] +"." + df_train["Slice"].astype(str)

# make image to feature dict
image_to_feature_train = {}
for i in range(len(feature_train)):
    image_to_feature_train[df_train["image_id"][i]] = i


# set seed
seed = 2022
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True

# hyperparameters
seq_len = 320
feature_size = 1280*3
lstm_size = 256
learning_rate = 0.001
batch_size = 64
num_epoch = 6

# dataset
df_labels = pd.read_csv(f'{RSNA_2022_PATH}/train.csv')
target_cols = ['patient_overall'] + [f'C{i}' for i in range(1, 8)]

df_labels["labels"] = df_labels[target_cols].apply(lambda x: x.values, axis=1)

series_to_labels = {}
for i, ID in enumerate(df_train["StudyInstanceUID"].unique()):
    series_to_labels[ID] = df_labels.loc[df_labels["StudyInstanceUID"]==ID]["labels"].values

class GRUDataSet(torch.utils.data.Dataset):
    def __init__(self,
                 feature_array,
                 image_to_feature,
                 series_to_labels,
                 series_dict,
                 image_dict,
                 series_list,
                 seq_len):
        self.feature_array=feature_array
        self.image_to_feature=image_to_feature
        self.series_to_labels=series_to_labels
        self.series_dict=series_dict
        self.image_dict=image_dict
        self.series_list=series_list # foldごとのlistを作る必要あり
        self.seq_len=seq_len
    def __len__(self):
        return len(self.series_list)
    def __getitem__(self,index):
        image_list = self.series_dict[self.series_list[index]]['sorted_image_list']
        y = self.series_to_labels[self.series_list[index]] # shape=(8,)
        if len(image_list)>self.seq_len:
            x = np.zeros((len(image_list), self.feature_array.shape[1]*3), dtype=np.float32)
            y_pe = np.zeros((len(image_list), 1), dtype=np.float32)
            mask = np.ones((self.seq_len,), dtype=np.float32)
            for i in range(len(image_list)):      
                x[i,:self.feature_array.shape[1]] = self.feature_array[self.image_to_feature[image_list[i]]] 
                #y_pe[i] = self.image_dict[image_list[i]]['pe_present_on_image']
            x = cv2.resize(x, (self.feature_array.shape[1]*3, self.seq_len), interpolation = cv2.INTER_LINEAR)
            #y_pe = np.squeeze(cv2.resize(y_pe, (1, self.seq_len), interpolation = cv2.INTER_LINEAR))
        else:
            x = np.zeros((self.seq_len, self.feature_array.shape[1]*3), dtype=np.float32)
            mask = np.zeros((self.seq_len,), dtype=np.float32)
            #y_pe = np.zeros((self.seq_len,), dtype=np.float32)
            for i in range(len(image_list)):      
                x[i,:self.feature_array.shape[1]] = self.feature_array[self.image_to_feature[image_list[i]]]
                mask[i] = 1.  
                #y_pe[i] = self.image_dict[image_list[i]]['pe_present_on_image']
        x[1:,self.feature_array.shape[1]:self.feature_array.shape[1]*2] = x[1:,:self.feature_array.shape[1]] - x[:-1,:self.feature_array.shape[1]]
        x[:-1,self.feature_array.shape[1]*2:] = x[:-1,:self.feature_array.shape[1]] - x[1:,:self.feature_array.shape[1]]
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.float32)
        return x, y, mask, self.series_list[index]

# model archetecture
class SpatialDropout(torch.nn.Dropout2d):
    def forward(self, x):
        x = x.unsqueeze(2)    # (N, T, 1, K)
        x = x.permute(0, 3, 2, 1)  # (N, K, 1, T)
        x = super(SpatialDropout, self).forward(x)  # (N, K, 1, T), some features are masked
        x = x.permute(0, 3, 2, 1)  # (N, T, 1, K)
        x = x.squeeze(2)  # (N, T, K)
        return x

class GRU(torch.nn.Module):
    def __init__(self, input_len, lstm_size):
        super().__init__()
        self.lstm1 = nn.GRU(input_len, lstm_size, bidirectional=True, batch_first=True)
        self.last_linear_frac = nn.Linear(lstm_size*2, 8)
        
    def forward(self, x, mask):
        x = SpatialDropout(0.3)(x)
        h_lstm1, _ = self.lstm1(x)
        #avg_pool = torch.mean(h_lstm2, 1)
        logits_frac = self.last_linear_frac(h_lstm1)
        logits_frac = torch.mean(logits_frac, 1)
        return logits_frac

# train functions

def save_model(name, model):
    torch.save(model.state_dict(), f'../output/{name}.pth')
    
def load_model(model, name, path='.'):
    data = torch.load(os.path.join(path, f'{name}.pth'), map_location=DEVICE)
    model.load_state_dict(data)
    return model

def gc_collect():
    gc.collect()
    torch.cuda.empty_cache()

def train_epoch(ep, model,  train_loader, logger, losses_meter, name, fold=None):
    
    model.train()
    scaler = GradScaler()
    with tqdm(train_loader, desc='Train', miniters=10) as progress:
        for batch_idx, (X, y, mask, index) in enumerate(progress):
            X = X.to(DEVICE)
            y = y.to(DEVICE)
            mask = mask.to(DEVICE)
            
            optimizer.zero_grad()
            # Using mixed precision training
            with autocast():
                y_frac_pred = model.forward(X, mask)
                frac_loss = weighted_loss(y_frac_pred, y)
                
                if np.isinf(frac_loss.item()) or np.isnan(frac_loss.item()):
                    print(f'Bad loss, skipping the batch {batch_idx}')
                    del frac_loss, y_frac_pred
                    gc_collect()
                    continue
                
                losses_meter.update(frac_loss.item(), X.size(0))
            
            scaler.scale(frac_loss).backward()
            scaler.step(optimizer)
            scaler.update()
        
            progress.set_description(f'Train loss: {frac_loss.item() :.02f}')

        #scheduler.step()

        logger.log({'train_frac_loss': losses_meter.avg, 'lr': scheduler.get_last_lr()[0]})

    save_model(name, model)
    return model, losses_meter.avg

def evaluate_epoch(model, eval_loader, logger, losses_meter):
    #model = GRU(input_len=feature_size, lstm_size=lstm_size).to(DEVICE)
    
    with torch.no_grad():
        model.eval()
        frac_losses = []
        pred_frac=[]
        with tqdm(eval_loader, desc='Eval', miniters=10) as progress:
            for i, (X, y, mask, index) in enumerate(progress):
                X = X.to(DEVICE)
                y = y.to(DEVICE)
                mask = mask.to(DEVICE)
                with autocast():
                    y_frac_pred = model.forward(X, mask)
                    frac_loss = weighted_loss(y_frac_pred, y)
                    pred_frac.append(torch.sigmoid(y_frac_pred))
                    frac_losses.append(frac_loss)
                
                losses_meter.update(frac_loss.item(), X.size(0))
                    
                    
        logger.log({'eval_frac_loss': losses_meter.avg})
        
    return losses_meter.avg, torch.concat(pred_frac).cpu().numpy()


if __name__ == '__main__':
    gru_models = []
for fold in range(1):
    if os.path.exists(os.path.join(GRU_CHECKPOINTS_PATH, f'gru_1-f{fold}.pth')):
        print(f'Found cached version of gru_1-f{fold}')
        gru_models.append(load_model(GRU(), f'gru_1-f{fold}', GRU_CHECKPOINTS_PATH))
    else:
        with wandb.init(project='RSNA-2022', name=f'GRU-1-fold{fold}_320_256_spatial_0.3_lr0.001_all_8000') as run:
            gc_collect()
            #series_list_train = list(df_train.query('split != @fold')["StudyInstanceUID"].unique())
            series_list_train = list(df_train["StudyInstanceUID"].unique())
            train_datagen = GRUDataSet(feature_array=feature_train,
                                        image_to_feature=image_to_feature_train,
                                        series_to_labels=series_to_labels,
                                        series_dict=series_dict,
                                        image_dict=image_dict,
                                        series_list=series_list_train,
                                        seq_len=seq_len)
            
            train_loader = torch.utils.data.DataLoader(train_datagen, batch_size=batch_size, shuffle=True,
                                                        num_workers=os.cpu_count(), pin_memory=True)
            
            
            model = GRU(input_len=feature_size, lstm_size=lstm_size).to(DEVICE)
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.8)

            for ep in range(num_epoch):
                train_loss_meter = AverageMeter()
                #valid_loss_meter = AverageMeter()
                model, loss = train_epoch(ep, model, train_loader, run, train_loss_meter, f'gru_1-all_8000', fold=fold)
                #eval_loss, frac_pred = evaluate_epoch(model, eval_loader, run, valid_loss_meter)    
                gru_models.append(model)
                
                scheduler.step()
                
                #print('epoch: {}, train_loss: {}, eval_loss: {}'.format(ep, loss, eval_loss))
                print('epoch: {}, train_loss: {}'.format(ep, loss))

