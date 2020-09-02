# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 10:12:09 2020

@author: Admin
"""

import soundfile as sf
import numpy as np
import pathlib
import os
import IPython
from utils import stft, istft
import tensorflow as tf
import matplotlib.pyplot as plt
from tf_wpe import wpe, online_wpe_step,get_power_online
from tqdm import tqdm


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
real_name='SimData_room2_near_female_ch1'
file_type='.wav'
file_name=real_name+file_type

signal_list,sampling_rate = sf.read(str(project_root / 'my_data' / file_name))

#Parameters
stft_options = dict(size=512, shift=128)
#Setup

channels = 1
delay = 3
alpha=0.99
taps = 10
frequency_bins = stft_options['size'] // 2 + 1

#
y = np.stack(signal_list, axis=0)
y = np.reshape(y,(1,y.shape[0]))
IPython.display.Audio(y[0], rate=sampling_rate)

Y = stft(y, **stft_options).transpose(1, 2, 0)
T, _, _ = Y.shape

Z_list = []

Q = np.stack([np.identity(channels * taps) for a in range(frequency_bins)])
G = np.zeros((frequency_bins, channels * taps, channels))

with tf.Session() as session:
    Y_tf = tf.placeholder(tf.complex128, shape=(taps + delay + 1, frequency_bins, channels))
    Q_tf = tf.placeholder(tf.complex128, shape=(frequency_bins, channels * taps, channels * taps))
    G_tf = tf.placeholder(tf.complex128, shape=(frequency_bins, channels * taps, channels))
    
    results = online_wpe_step(Y_tf, get_power_online(tf.transpose(Y_tf, (1, 0, 2))), Q_tf, G_tf, alpha=alpha, taps=taps, delay=delay)
    for Y_step in tqdm(aquire_framebuffer(Y,T,taps,delay)):
        feed_dict = {Y_tf: Y_step, Q_tf: Q, G_tf: G}
        Z, Q, G = session.run(results, feed_dict)
        Z_list.append(Z)

Z_stacked = np.stack(Z_list)
z = istft(np.asarray(Z_stacked).transpose(2, 0, 1), size=stft_options['size'], shift=stft_options['shift'])

IPython.display.Audio(z[0], rate=sampling_rate)

#Power Spectrum
fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(20, 8))
im1 = ax1.imshow(20 * np.log10(np.abs(Y[:, :, 0])).T, origin='lower')
ax1.set_xlabel('')
_ = ax1.set_title('reverberated')
im2 = ax2.imshow(20 * np.log10(np.abs(Z_stacked[:, :, 0])).T, origin='lower')
_ = ax2.set_title('dereverberated')
cb = fig.colorbar(im1)

#save the dereverberted sound file
z_save=np.reshape(z,(z.shape[1],))
outputname=real_name+'_online_fix'+file_type
sf.write(str(project_root / 'output' / outputname), z_save, sampling_rate)
