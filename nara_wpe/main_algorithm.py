# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 16:31:47 2020

@author: Admin
"""

import IPython
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import time
from tqdm import tqdm

import pathlib
import os

from nara_wpe.tf_wpe import wpe
from nara_wpe.wpe import get_power
from nara_wpe.wpe import online_wpe_step, get_power_online, OnlineWPE
from nara_wpe.utils import stft, istft, get_stft_center_frequencies

def offline_algorithm(signal_list,sampling_rate,stft_options,
                      channels,delay,iterations,taps,alpha,
                      project_root,file_name):
    y = np.stack(signal_list, axis=0)
    y = np.reshape(y,(1,y.shape[0]))
    IPython.display.Audio(y[0], rate=sampling_rate)

    Y = stft(y, **stft_options).transpose(2, 0, 1)

    Z = wpe(
          Y,
         taps=taps,
         delay=delay,
         iterations=iterations,
         statistics_mode='full'
          ).transpose(1, 2, 0)
    z = istft(Z, size=stft_options['size'], shift=stft_options['shift'])
    IPython.display.Audio(z[0], rate=sampling_rate)
    #save the dereverbration sound file
    z_save=np.reshape(z,(z.shape[1],))
    outputname= 'offline_fix_'+file_name
    sf.write(str(project_root / 'my_data' /'0.2 - 0.5'/'offline_fix'/ outputname), z_save, sampling_rate)
    print("Finish Offline Algorithm")
    
def aquire_framebuffer(Y,T,taps,delay):
    buffer = list(Y[:taps+delay+1, :, :])
    for t in range(taps+delay+1, T):
        yield np.array(buffer)
        buffer.append(Y[t, :, :])
        buffer.pop(0)

def online_algorithm(signal_list,sampling_rate,stft_options,
                      channels,delay,taps,alpha,frequency_bins,
                      project_root,file_name):
    y = np.stack(signal_list, axis=0)
    y = np.reshape(y,(1,y.shape[0]))
    Y = stft(y, **stft_options).transpose(1, 2, 0)
    T, _, _ = Y.shape

    #Non-iterative frame online approach
    Z_list = []
    Q = np.stack([np.identity(channels * taps) for a in range(frequency_bins)])
    G = np.zeros((frequency_bins, channels * taps, channels))

    for Y_step in tqdm(aquire_framebuffer(Y,T,taps,delay)):
        Z, Q, G = online_wpe_step(
                  Y_step,
                  get_power_online(Y_step.transpose(1, 2, 0)),
                  Q,
                  G,
                  alpha=alpha,
                  taps=taps,
                  delay=delay)
        Z_list.append(Z)

    Z_stacked = np.stack(Z_list)
    z = istft(np.asarray(Z_stacked).transpose(2, 0, 1), size=stft_options['size'], shift=stft_options['shift'])

    IPython.display.Audio(z[0], rate=sampling_rate)
    #save the dereverberted sound file
    z_save=np.reshape(z,(z.shape[1],))
    outputname='online_fix_'+file_name
    sf.write(str(project_root / 'my_data' / '0.2 - 0.5' /'online_fix'/ outputname), z_save, sampling_rate)
    print("Finish Online Algorithm")
    
if __name__=='__main__':
    #Parameters
    stft_options = dict(size=512, shift=128)
    #Setup
    channels = 1
    delay = 3
    alpha=0.99
    taps = 10
    frequency_bins = stft_options['size'] // 2 + 1
    iterations=5
    project_root = pathlib.Path(os.path.abspath(
         os.path.join(os.path.dirname(__file__), os.pardir)))

    filesFromDir = os.listdir(str(project_root / 'my_data'/'0.2 - 0.5'/'reverb_data' ))  
    for file_name in filesFromDir:
        signal_list,sampling_rate = sf.read(str(project_root / 'my_data' /'0.2 - 0.5'/ 'reverb_data'/ file_name))
        online_algorithm(signal_list,sampling_rate,stft_options,
                      channels,delay,taps,alpha,frequency_bins,
                      project_root,file_name)
        offline_algorithm(signal_list,sampling_rate,stft_options,
                      channels,delay,iterations,taps,alpha,
                      project_root,file_name)