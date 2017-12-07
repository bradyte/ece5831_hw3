#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 17:05:07 2017

@author: tbrady
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from hmmlearn import hmm
from python_speech_features import mfcc
from scipy.io import wavfile


###############################################################################
### https://github.com/yashiro32/speech_recognition/blob/master/speech_rec_w_essentia.py~
###############################################################################

# get data
fpaths = []
labels = []
spoken = []
for f in os.listdir('trainingSet'):
    if not f.startswith('.'):
        for w in os.listdir('trainingSet/' + f):
            fpaths.append('trainingSet/' + f + '/' + w)
            labels.append(f)
            if f not in spoken:
                spoken.append(f)
            
sampleRate  = 16000 # wav file sample rate
bitRate     = 16 # wav file bit rate

data = np.zeros((len(fpaths),int(sampleRate * bitRate / 8)))
maxsize = -1
for n,file in enumerate(fpaths):
    rate, d = wavfile.read(file)
    data[n, :d.shape[0]] = d
    if d.shape[0] > maxsize:
        maxsize = d.shape[0]
data = data[:, :maxsize]

#all_labels = np.zeros(data.shape[0])
#for n, l in enumerate(set(labels)):
#    all_labels[np.array([i for i, _ in enumerate(labels) if _ == l])] = n
#print ('Labels and label indices',all_labels)


###############################################################################
### http://www.practicalcryptography.com/miscellaneous/machine-learning/guide-mel-frequency-cepstral-coefficients-mfccs/
### http://practicalcryptography.com/miscellaneous/machine-learning/hidden-markov-model-hmm-tutorial/
###############################################################################
idx = 2


plt.figure()
mfcc_feat = mfcc(data[idx, :])
plt.subplot(211)
Pxx, freqs, bins, im = plt.specgram(data[idx, :], NFFT=256, Fs=2, noverlap=128)

plt.subplot(212)
plt.plot(data[idx, :])

#plt.title('Timeseries example for %s'%labels[idx])
#plt.xlim(0, maxsize)
#plt.xlabel('Time (samples)')
#plt.ylabel('Amplitude (signed 16 bit)')

plt.figure()
for i in range(1,len(mfcc_feat[1])):
    plt.subplot(2,13,i+1)
    plt.plot(mfcc_feat[:,i])
    plt.ylim([-50,50])
    plt.axis('off') 
plt.figure()
for idx in range(10,20):
    mfcc_feat = mfcc(data[idx, :])
    plt.plot(np.mean(mfcc_feat[:,1:], axis=0))
    plt.ylim([-30,30])
plt.show()






#model = hmm.GaussianHMM(n_components=10, covariance_type="full")












