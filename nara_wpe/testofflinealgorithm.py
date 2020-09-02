# -*- coding: utf-8 -*-
"""
Created on Sun Jul 19 11:09:43 2020

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
from tf_wpe import wpe, get_power

project_root = pathlib.Path(os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir)
))

#Audio Data
real_name='SimData_room2_near_female_ch1'
file_type='.wav'
file_name=real_name + file_type
hey=str(project_root / 'my_data' / file_name)

signal_list,sampling_rate = sf.read(str(project_root / 'my_data' / file_name))

#Parameters
stft_options = dict(size=512, shift=128)
#Setup
channels = 1
delay = 3
iterations = 5
taps = 10


y = np.stack(signal_list, axis=0)
y = np.reshape(y,(1,y.shape[0]))
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
im1 = ax1.imshow(20 * np.log10(np.abs(Y[:, 0, :])), origin='lower')
ax1.set_xlabel('')
_ = ax1.set_title('reverberated')
im2 = ax2.imshow(20 * np.log10(np.abs(Z[:, 0, :])), origin='lower')
_ = ax2.set_title('dereverberated')
cb = fig.colorbar(im1)

#save the dereverbration sound file
z_save=np.reshape(z,(z.shape[1],))
outputname=real_name + '_offline_fix'+file_type
sf.write(str(project_root / 'output' / outputname), z_save, sampling_rate)


