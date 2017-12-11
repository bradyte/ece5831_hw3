#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 10:57:35 2017

@author: tbrady
"""


import numpy as np
import matplotlib.pyplot as plt

def printTimeDomainWaveform(data, label, fpaths):
    time_units = np.arange(0,1,1/16000)
    
    plt.figure(figsize=(10, 4))
    
    wave = plt.plot(time_units, data, label = label, color='g')
    plt.title('{} waveform from {}'.format(label, fpaths))
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude (32-bit Audio)')
    plt.ylim([-2**15,2**15]) # 32-bit audio
    plt.grid('on')
    plt.legend(handles = wave)