#!/usr/bin/env python
# coding: utf-8

# In[3]:


import os
import numpy as np
np.random.seed(1969)
import tensorflow as tf
tf.set_random_seed(1969)


from scipy import signal
from glob import glob
import re
import pandas as pd
import gc
from scipy.io import wavfile

from keras import optimizers, losses, activations, models
from keras.layers import GRU, Convolution2D, Dense, Input, Flatten, Dropout, MaxPooling2D, BatchNormalization, Conv3D, ConvLSTM2D,Conv1D,Activation,LSTM
from keras.callbacks import TensorBoard
from keras.models import Sequential
from tqdm import tqdm
from sklearn.model_selection import GroupKFold
from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import logfbank
import random
import os
import pandas as pd
import librosa
import glob
import torch
from torch import *
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


# In[46]:


import argparse
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str,
    help="path to test file")
args = ap.parse_args()


# In[4]:


class ConvBn2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, dilation=1, stride=1, groups=1, is_bn=True):
        super(ConvBn2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups, bias=False)
        self.bn   = nn.BatchNorm2d(out_channels)
        if is_bn is False:
            self.bn =None

    def forward(self,x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        return x

class SeScale(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SeScale, self).__init__()
        self.fc1 = nn.Conv2d(channel, reduction, kernel_size=1, padding=0)
        self.fc2 = nn.Conv2d(reduction, channel, kernel_size=1, padding=0)

    def forward(self, x):
        x = F.adaptive_avg_pool2d(x,1)
        x = self.fc1(x)
        x = F.relu(x, inplace=True)
        x = self.fc2(x)
        x = F.sigmoid(x)
        return x


class ResBlock(nn.Module):
    def __init__(self, in_planes, out_planes, reduction=16):
        super(ResBlock, self).__init__()
        assert(in_planes==out_planes)

        self.conv_bn1 = ConvBn2d(in_planes,  out_planes, kernel_size=3, padding=1, stride=1)
        self.conv_bn2 = ConvBn2d(out_planes, out_planes, kernel_size=3, padding=1, stride=1)
        self.scale    = SeScale(out_planes, reduction)

    def forward(self, x):
        z  = F.relu(self.conv_bn1(x),inplace=True)
        z  = self.conv_bn2(z)
        z  = self.scale(z)*z + x
        z  = F.relu(z,inplace=True)
        return z



## net ##-------

class SeResNet3(nn.Module):
    def __init__(self, in_shape=(1,40,101), num_classes=5 ):
        super(SeResNet3, self).__init__()
        in_channels = in_shape[0]

        self.layer1a = ConvBn2d(in_channels, 16, kernel_size=(3, 3), stride=(1, 1))
        self.layer1b = ResBlock( 16, 16)

        self.layer2a = ConvBn2d(16, 32, kernel_size=(3, 3), stride=(1, 1))
        self.layer2b = ResBlock(32, 32)
        self.layer2c = ResBlock(32, 32)

        self.layer3a = ConvBn2d(32, 64, kernel_size=(3, 3), stride=(1, 1))
        self.layer3b = ResBlock(64, 64)
        self.layer3c = ResBlock(64, 64)

        self.layer4a = ConvBn2d( 64,128, kernel_size=(3, 3), stride=(1, 1))
        self.layer4b = ResBlock(128,128)
        self.layer4c = ResBlock(128,128)

        self.layer5a = ConvBn2d(128, 256, kernel_size=(3, 3), stride=(1, 1))
        self.layer5b = nn.Linear(256,256)

        self.fc = nn.Linear(256,num_classes)


    def forward(self, x):

        x = F.relu(self.layer1a(x),inplace=True)
        x = self.layer1b(x)
        x = F.max_pool2d(x,kernel_size=(2,2),stride=(2,2))

        x = F.dropout(x,p=0.1,training=self.training)
        x = F.relu(self.layer2a(x),inplace=True)
        x = self.layer2b(x)
        x = self.layer2c(x)
        x = F.max_pool2d(x,kernel_size=(2,2),stride=(2,2))

        x = F.dropout(x,p=0.2,training=self.training)
        x = F.relu(self.layer3a(x),inplace=True)
        x = self.layer3b(x)
        x = self.layer3c(x)
        x = F.max_pool2d(x,kernel_size=(2,2),stride=(2,2))

        x = F.dropout(x,p=0.2,training=self.training)
        x = F.relu(self.layer4a(x),inplace=True)
        x = self.layer4b(x)
        x = self.layer4c(x)

        x = F.dropout(x,p=0.2,training=self.training)
        x = F.relu(self.layer5a(x),inplace=True)
        x = F.adaptive_avg_pool2d(x,1)
        x = x.view(x.size(0), -1)
        x = F.relu(self.layer5b(x))

        x = F.dropout(x,p=0.2,training=self.training)
        x = self.fc(x)

        return x  #logits


# In[6]:


H = 40
W = 101
modeltest = SeResNet3(in_shape=(1,H,W), num_classes=5).cuda()
modeltest.load_state_dict(torch.load("model.pth"))
modeltest.eval()


# In[7]:


from torch.utils.data import Dataset, DataLoader
class MyDataset(Dataset):
    def __init__(self,dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        data, target = self.dataset[index]

        return data, target, index
    def __len__(self):
        return len(self.dataset)


# In[8]:


mappings={"disgust":int(0),"fear":int(1),"happy":int(2),"neutral":int(3),"sad":int(4)}


# In[10]:


AUDIO_LENGTH=16000
AUDIO_SR=16000
def tf_random_add_noise_transform(wave, noise_limit=0.2, u=0.5):

    if random.random() < u:
        num_noises = len(AUDIO_NOISES)
        noise = AUDIO_NOISES[np.random.choice(num_noises)]

        wave_length  = len(wave)
        noise_length = len(noise)
        p=noise_length - wave_length - 1
        print(p)
        t = np.random.randint(0, noise_length - wave_length - 1)
        #t = np.random.randint(noise_length - wave_length - 1,0)
        noise = noise[t:t + wave_length]

        alpha = np.random.random() * noise_limit
        wave  = np.clip(alpha * noise + wave, -1, 1)

    return wave


def tf_random_time_shift_transform(wave, shift_limit=0.2, u=0.5):
    if random.random() < u:
        wave_length  = len(wave)
        shift_limit = shift_limit*wave_length
        shift = np.random.randint(-shift_limit, shift_limit)
        t0 = -min(0, shift)
        t1 =  max(0, shift)
        wave = np.pad(wave, (t0, t1), 'constant')
        wave = wave[:-t0] if t0 else wave[t1:]

    return wave


def tf_random_pad_transform(wave, length=AUDIO_LENGTH):

    if len(wave)<AUDIO_LENGTH:
        L = abs(len(wave)-AUDIO_LENGTH)
        start = np.random.choice(L)
        wave  = np.pad(wave, (start, L-start), 'constant')

    elif len(wave)>AUDIO_LENGTH:
        L = abs(len(wave)-AUDIO_LENGTH)
        start = np.random.choice(L)
        wave  = wave[start: start+AUDIO_LENGTH]

    return wave


def tf_fix_pad_transform(wave, length=AUDIO_LENGTH):
    # wave = np.pad(wave, (0, max(0, AUDIO_LENGTH - len(wave))), 'constant')
    # return wave

    if len(wave)<AUDIO_LENGTH:
        L = abs(len(wave)-AUDIO_LENGTH)
        start = L//2
        wave  = np.pad(wave, (start, L-start), 'constant')

    elif len(wave)>AUDIO_LENGTH:
        L = abs(len(wave)-AUDIO_LENGTH)
        start = L//2
        wave  = wave[start: start+AUDIO_LENGTH]

    return wave




def tf_random_scale_amplitude_transform(wave, scale_limit=0.1, u=0.5):
    if random.random() < u:
        scale = np.random.randint(-scale_limit, scale_limit)
        wave = scale*wave
    return wave
##  mfcc ,spectrogram ####################################################################

def tf_wave_to_mfcc(wave):

    spectrogram = librosa.feature.melspectrogram(wave, sr=AUDIO_SR, n_mels=40, hop_length=160, n_fft=480, fmin=20, fmax=4000)
    #spectrogram = librosa.power_to_db(spectrogram)
    idx = [spectrogram > 0]
    spectrogram[idx] = np.log(spectrogram[idx])

    dct_filters = librosa.filters.dct(n_filters=40, n_input=40)
    mfcc = [np.matmul(dct_filters, x) for x in np.split(spectrogram, spectrogram.shape[1], axis=1)]
    mfcc = np.hstack(mfcc)
    mfcc = mfcc.astype(np.float32)

    return mfcc


def tf_wave_to_melspectrogram(wave):
    spectrogram = librosa.feature.melspectrogram(wave, sr=AUDIO_SR, n_mels=40, hop_length=160, n_fft=480, fmin=20, fmax=4000)
    spectrogram = librosa.power_to_db(spectrogram)
    spectrogram = spectrogram.astype(np.float32)

    return spectrogram



def tf_wave_to_melspectrogram_mfcc(wave):

    spectrogram = librosa.feature.melspectrogram(wave, sr=AUDIO_SR, n_mels=40, hop_length=160, n_fft=480, fmin=5, fmax=4500)
    idx = [spectrogram > 0]
    spectrogram[idx] = np.log(spectrogram[idx])

    dct_filters = librosa.filters.dct(n_filters=40, n_input=40)
    mfcc = [np.matmul(dct_filters, x) for x in np.split(spectrogram, spectrogram.shape[1], axis=1)]
    mfcc = np.hstack(mfcc)
    mfcc = mfcc.astype(np.float32)

    spectrogram = librosa.power_to_db(spectrogram)
    spectrogram = spectrogram.astype(np.float32)

    all = np.concatenate((spectrogram[np.newaxis,:],mfcc[np.newaxis,:]))
    return all




##--------------
def tf_wave_to_melspectrogram1(wave):
    spectrogram = librosa.feature.melspectrogram(wave, sr=AUDIO_SR, n_mels=40, hop_length=160, n_fft=480, fmin=20, fmax=4000)
    idx = [spectrogram > 0]
    spectrogram[idx] = np.log(spectrogram[idx])
    spectrogram = spectrogram.astype(np.float32)
    return spectrogram


# In[11]:


def train_augment(wave):
    wave = tf_random_time_shift_transform(wave, shift_limit=0.2, u=0.5)
#    wave = tf_random_add_noise_transform (wave, noise_limit=0.2, u=0.5)
    wave = tf_random_pad_transform(wave)

    tensor = tf_wave_to_melspectrogram(wave)[np.newaxis,:]
    #tensor = tf_wave_to_mfcc(wave)[np.newaxis,:]
    #tensor = torch.from_numpy(tensor)
    return tensor


def valid_augment(wave):
    wave = tf_fix_pad_transform(wave)

    tensor = tf_wave_to_melspectrogram(wave)[np.newaxis,:]
    #tensor = tf_wave_to_mfcc(wave)[np.newaxis,:]
    #tensor = torch.from_numpy(tensor)
    return tensor


# In[38]:


# new sampling
new_sample_rate = 16000
y_test = []
x_test = []
names=[]
inputfolder=args.input

mylist1=os.listdir(inputfolder)

for file in mylist1:
    samples, sample_rate = librosa.core.load(inputfolder+"/"+file,mono=True,sr=16000)
    specgram=valid_augment(samples)

    x_test.append(specgram)
    y_test.append(0)
    names.append(file)
        
x_test = np.array(x_test)
y_test=np.array(y_test)



# In[39]:



x_test = torch.Tensor(x_test).cuda()
y_test = torch.cuda.LongTensor(y_test).cuda()
valid = torch.utils.data.TensorDataset(x_test, y_test)
valid = MyDataset(valid)

valid_loader = torch.utils.data.DataLoader(valid, batch_size=128, shuffle=False)


# In[40]:


predictions=[]
for i, (x_batch, y_batch, index) in enumerate(valid_loader):
    y_pred = modeltest(x_batch).detach()
    prediction = torch.argmax(y_pred, dim=1)
    
    predictions.append(prediction.tolist())


# In[41]:


flattened_list = [y for x in predictions for y in x]


# In[42]:


mappings1={0:"disgust",1:"fear",2:"happy",3:"neutral",4:"sad"}


# In[43]:


for i in range(len(flattened_list)):
    if flattened_list[i] in mappings1:
        flattened_list[i]=mappings1[flattened_list[i]]


# In[29]:


submission=pd.DataFrame()
submission["File name"]=names
submission["prediction"]=flattened_list
submission.to_csv("Submission.txt",sep=',', index=False)
#submission.to_csv("Submission.csv",sep=',', index=False)


