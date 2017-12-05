#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 17:05:07 2017

@author: tbrady
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import os
import scipy
import math
from scipy.io import wavfile


fpaths = []
labels = []
spoken = []

def stft(x, fftsize=64, overlap_pct=.5):   
    #Modified from http://stackoverflow.com/questions/2459295/stft-and-istft-in-python
    hop = int(fftsize * (1 - overlap_pct))
    w = scipy.hanning(fftsize + 1)[:-1]    
    raw = np.array([np.fft.rfft(w * x[i:i + fftsize]) for i in range(0, len(x) - fftsize, hop)])
    return raw[:, :(fftsize // 2)]

path = 'trainingSet'
for f in os.listdir(path):
    if not f.startswith('.'):
        for w in os.listdir(path + '/' + f):
            fpaths.append(path + '/' + f + '/' + w)
            labels.append(f)
            if f not in spoken:
                spoken.append(f)
            
print('Words Spoken:', spoken)


sampleRate  = 16000 # wav file sample rate
bitRate     = 16 # wav file bit rate
fileSize    = int(sampleRate * bitRate / 8) # file size in bytes

data = np.zeros((len(fpaths),fileSize))
maxsize = -1
for n,file in enumerate(fpaths):
    _, d = wavfile.read(file)
    data[n, :d.shape[0]] = d
    if d.shape[0] > maxsize:
        maxsize = d.shape[0]
data = data[:, :maxsize]

#print('Number of files total:', data.shape[0])
#all_labels = np.zeros(data.shape[0])
#for n, l in enumerate(set(labels)):
#    all_labels[np.array([i for i, _ in enumerate(labels) if _ == l])] = n
    
#print('Labels and label indices', all_labels)
idx = 0
plt.plot(data[idx, :], color='steelblue')
plt.title('Timeseries example for %s'%labels[idx])
plt.xlim(0, maxsize)
plt.xlabel('Time (samples)')
plt.ylabel('Amplitude (signed 16 bit)')
plt.figure()

# + 1 to avoid log of 0
log_freq = 20 * np.log(np.abs(stft(data[0, :])) + 1)
print(log_freq.shape)
plt.imshow(log_freq, cmap='gray', interpolation=None)
plt.xlabel('Freq (bin)')
plt.ylabel('Time (overlapped frames)')
plt.ylim(log_freq.shape[1])
plt.title('PSD of %s example'%labels[0])
