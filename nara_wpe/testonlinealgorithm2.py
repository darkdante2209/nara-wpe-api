# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 11:23:20 2020

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

from wpe import online_wpe_step, get_power_online, OnlineWPE
from utils import stft, istft, get_stft_center_frequencies

def aquire_framebuffer(Y,T,taps,delay):
    buffer = list(Y[:taps+delay+1, :, :])
    for t in range(taps+delay+1, T):
        yield np.array(buffer)
        buffer.append(Y[t, :, :])
        buffer.pop(0)
        
project_root = pathlib.Path(os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir)
))

#Audio Data
real_name='reverb_Nguyen_Doan_Toan _PCT'
file_type='.wav'
file_name=real_name + file_type

signal_list,sampling_rate = sf.read(str(project_root / 'my_data' / '0.1 - 0.7' / 'reverb_data'/ file_name))

y = np.stack(signal_list, axis=0)
y = np.reshape(y,(1,y.shape[0]))
IPython.display.Audio(y[0], rate=sampling_rate)
#Parameters
stft_options = dict(size=512, shift=128)
#Setup
channels = 1
delay = 3
alpha=0.99
taps = 10
frequency_bins = stft_options['size'] // 2 + 1

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
        delay=delay
    )
    Z_list.append(Z)

Z_stacked = np.stack(Z_list)
z = istft(np.asarray(Z_stacked).transpose(2, 0, 1), size=stft_options['size'], shift=stft_options['shift'])

IPython.display.Audio(z[0], rate=sampling_rate)

# Frame online WPE in class fashion:
# Z_list = []
# online_wpe = OnlineWPE(
#     taps=taps,
#     delay=delay,
#     alpha=alpha
# )
# for Y_step in tqdm(aquire_framebuffer(Y,T,taps,delay)):
#     Z_list.append(online_wpe.step_frame(Y_step))

# Z_stacked = np.stack(Z_list)
# z = istft(np.asarray(Z_stacked).transpose(2, 0, 1), size=stft_options['size'], shift=stft_options['shift'])

# IPython.display.Audio(z[0], rate=sampling_rate)

# Power spectrum
fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(20, 8))
im1 = ax1.imshow(20 * np.log10(np.abs(Y[200:400, :, 0])).T, origin='lower')
ax1.set_xlabel('')
_ = ax1.set_title('reverberated')
im2 = ax2.imshow(20 * np.log10(np.abs(Z_stacked[200:400, :, 0])).T, origin='lower')
_ = ax2.set_title('dereverberated')
cb = fig.colorbar(im1)

#save the dereverberted sound file
z_save=np.reshape(z,(z.shape[1],))
outputname=real_name+'_online_fix'+file_type
sf.write(str(project_root / 'my_data' /'online_fix'/ outputname), z_save, sampling_rate)

#plot signal
plt.figure(2)
axs1 = plt.subplot(211)
plt.plot(y[0,:])
plt.subplot(212, sharex=axs1, sharey=axs1)
plt.plot(z[0,:])
plt.show()