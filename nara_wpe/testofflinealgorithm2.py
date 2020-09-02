# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 11:23:22 2020

@author: Admin
"""
import IPython
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from tqdm import tqdm

import pathlib
import os

from wpe import wpe
from wpe import get_power
from utils import stft, istft, get_stft_center_frequencies

project_root = pathlib.Path(os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir)
))

#Audio Data
real_name='reverb_Nguyen_Doan_Toan _PCT'
file_type='.wav'
file_name=real_name + file_type

signal_list,sampling_rate = sf.read(str(project_root / 'my_data' / '0.1 - 0.7'/ 'reverb_data'/ file_name))

#Parameters
stft_options = dict(size=512, shift=128)
#Setup
channels = 1
delay = 3
iterations = 5
taps = 10
alpha=0.9999


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
print("Finish dereverberation")
#Power spectrum
# plt.figure(1)
# fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(20, 10))
# im1 = ax1.imshow(20 * np.log10(np.abs(Y[ :, 0, 200:400])), origin='lower')
# ax1.set_xlabel('frames')
# _ = ax1.set_title('reverberated')
# im2 = ax2.imshow(20 * np.log10(np.abs(Z[0, 200:400, :])).T, origin='lower', vmin=-120, vmax=0)
# ax2.set_xlabel('frames')
# _ = ax2.set_title('dereverberated')
# cb = fig.colorbar(im2)

#save the dereverbration sound file
z_save=np.reshape(z,(z.shape[1],))
outputname=real_name + '_offline_fix'+file_type
sf.write(str(project_root / 'output' / outputname), z_save, sampling_rate)

#plot signal
plt.figure(2)
axs1 = plt.subplot(211)
plt.plot(y[0,:])
plt.subplot(212, sharex=axs1, sharey=axs1)
plt.plot(z[0,:])
plt.show()

print(y[0,:].shape)
print(z[0,:].shape)