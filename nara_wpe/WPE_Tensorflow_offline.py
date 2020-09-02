# -*- coding: utf-8 -*-
"""
Created on Sat Jul 18 11:02:16 2020

@author: Admin
"""
import IPython
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from tqdm import tqdm
import tensorflow as tf

from tf_wpe import wpe, get_power
from utils import stft, istft, get_stft_center_frequencies


import pathlib
import os

def aquire_audio_data():
    D, T = 4, 10000
    y = np.random.normal(size=(D, T))
    return y

project_root = pathlib.Path(os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir)
))

#Minimal example with random data
stft_options = dict(size=512, shift=128)
y = aquire_audio_data()
Y = stft(y, **stft_options).transpose(2, 0, 1)
with tf.Session() as session:
    Y_tf = tf.placeholder(
        tf.complex128, shape=(None, None, None))
    Z_tf = wpe(Y_tf)
    Z = session.run(Z_tf, {Y_tf: Y})
z_tf = istft(Z.transpose(1, 2, 0), size=stft_options['size'], shift=stft_options['shift'])

#Example with real audio recording
#Setup
channels = 8
sampling_rate = 16000
delay = 3
iterations = 5
taps = 10
#Audio Data
file_template = 'AMI_WSJ20-Array1-{}_T10c0201.wav'
signal_list = [
    sf.read(str(project_root / 'data' / file_template.format(d + 1)))[0]
    for d in range(channels)
]
y = np.stack(signal_list, axis=0)
IPython.display.Audio(y[0], rate=sampling_rate)
Y = stft(y, **stft_options).transpose(2, 0, 1)
with tf.Session()as session:
    Y_tf = tf.placeholder(tf.complex128, shape=(None, None, None))
    Z_tf = wpe(Y_tf, taps=taps, iterations=iterations)
    Z = session.run(Z_tf, {Y_tf: Y})
z = istft(Z.transpose(1, 2, 0), size=stft_options['size'], shift=stft_options['shift'])
IPython.display.Audio(z[0], rate=sampling_rate)
#Power Spectrum
fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(20, 8))
im1 = ax1.imshow(20 * np.log10(np.abs(Y[:, 0, 200:400])), origin='lower')
ax1.set_xlabel('')
_ = ax1.set_title('reverberated')
im2 = ax2.imshow(20 * np.log10(np.abs(Z[:, 0, 200:400])), origin='lower')
_ = ax2.set_title('dereverberated')
cb = fig.colorbar(im1)

