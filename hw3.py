#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 17:05:07 2017

@author: tbrady
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
from matplotlib import cm
import time
import os
from hmmlearn import hmm
from python_speech_features import mfcc, logfbank
from scipy.io import wavfile
import warnings

import operator

warnings.filterwarnings("ignore")

tsys = time.time()
###############################################################################
'''
Get the filepaths of the wav files
https://github.com/yashiro32/speech_recognition/blob/master/speech_rec_w_essentia.py~
'''
###############################################################################
fpaths = []
labels = []
spoken = []
sample_rate = 16000

for f in os.listdir('trainingSet'):
    if not f.startswith('.'):
        for w in os.listdir('trainingSet/' + f):
            if not w.startswith('.'):
                fpaths.append('trainingSet/' + f + '/' + w)
                labels.append(f)
                if f not in spoken:
                    spoken.append(f)

###############################################################################
'''
Read the data into the program

Find the filterbanks
http://www.practicalcryptography.com/miscellaneous/machine-learning/guide-mel-frequency-cepstral-coefficients-mfccs/

Find the MFCCs
http://www.practicalcryptography.com/miscellaneous/machine-learning/guide-mel-frequency-cepstral-coefficients-mfccs/

'''
###############################################################################    
data        = np.zeros((len(fpaths),sample_rate))
fbank_feats = []
mfcc_feats  = []
maxsize     = -1
lengths     = []
for n, file in enumerate(fpaths):
    sample_rate, d       = wavfile.read(file)
    data[n, :d.shape[0]] = d
    if d.shape[0] > maxsize:
        maxsize   = d.shape[0]
    fbank_feats.append(logfbank(d,samplerate=16000,winlen=0.025,winstep=0.01,
                nfilt=40,nfft=512,preemph=0.97))
    mfcc_feats.append(mfcc(d,samplerate=16000,winlen=0.025,winstep=0.01,numcep=13,
                nfilt=40,nfft=512,preemph=0.97,ceplifter=22,appendEnergy=True))
    lengths.append(len(mfcc_feats[n]))
data    = data[:, :maxsize]
idx     = 0
X = np.concatenate(mfcc_feats)

###############################################################################
'''
Run the HMM
http://practicalcryptography.com/miscellaneous/machine-learning/hidden-markov-model-hmm-tutorial/
'''
############################################################################### 
print('Beginning HMM...')
model = hmm.GaussianHMM(n_components=10, covariance_type='full').fit([X], lengths)

#tfpaths = []
#tlabels = []
#tspoken = []
#sample_rate = 16000
#
#for f in os.listdir('testingSet'):
#    if not f.startswith('.'):
#        for w in os.listdir('testingSet/' + f):
#            if not w.startswith('.'):
#                tfpaths.append('testingSet/' + f + '/' + w)
#                tlabels.append(f)
#                if f not in tspoken:
#                    tspoken.append(f)
#
#tdata        = np.zeros((len(tfpaths),sample_rate))
#tfbank_feats = []
#tmfcc_feats  = []
#maxsize     = -1
#for n, file in enumerate(tfpaths):
#    sample_rate, td       = wavfile.read(file)
#    tdata[n, :td.shape[0]] = td
#    if td.shape[0] > maxsize:
#        maxsize   = td.shape[0]
#    tfbank_feats.append(logfbank(td,samplerate=16000,winlen=0.025,winstep=0.01,
#                nfilt=40,nfft=512,preemph=0.97))
#    tmfcc_feats.append(mfcc(td,samplerate=16000,winlen=0.025,winstep=0.01,numcep=13,
#                nfilt=40,nfft=512,preemph=0.97,ceplifter=22,appendEnergy=True))
#    
#tdata    = tdata[:, :maxsize]



#guessIndex = np.zeros(10)
#correct = 0
#ik = 90
#for i in range(ik,ik+10):
#    guess = model.score_samples(tmfcc_feats[i])[1]
#    guessSum = np.sum(guess, axis=0)
#    index, value = max(enumerate(guessSum), key=operator.itemgetter(1))
#    guessIndex[index] = guessIndex[index] + 1
#
#    
##    print('{}\t{}'.format(tlabels[i],index))
#index, value = max(enumerate(guessIndex), key=operator.itemgetter(1))
#plt.figure(figsize=(2,2))
#plt.bar(np.arange(10), np.array(guessIndex))
#plt.xlabel('Class Guess = {}\nCorrect Class = {}'.format(index, str(ik)[0]))
#plt.title(tlabels[i])
#
#plt.figure(figsize=(10, 4))
#wave = plt.plot(np.arange(0,1,1/16000), data[i,:], label = labels[idx], color='g')
#plt.title('{} waveform from {}'.format(labels[idx], fpaths[idx]))
#plt.xlabel('Time (s)')
#plt.ylabel('Amplitude (32-bit Audio)')
#plt.ylim([-2**15,2**15]) # 32-bit audio
#plt.grid('on')
#plt.legend(handles = wave)
#
#plt.figure(figsize=(10, 4))
#pylab.imshow(fbank_feats[idx].T, origin='lower',aspect='auto', cmap='jet')
#plt.title('{} filterbank from {}'.format(labels[idx], fpaths[idx]))
#plt.ylabel('Filterbanks')
#plt.xlabel('Time (10ms)')
#
#plt.figure(figsize=(10, 4))
#pylab.imshow(mfcc_feats[idx].T, origin='lower',aspect='auto', cmap='jet')
#plt.title('{} MFCCs from {}'.format(labels[idx], fpaths[idx]))
#plt.ylabel('MFCCs')
#plt.xlabel('Time (10ms)')



print('\nExecution time: {}'.format(time.time() - tsys))


