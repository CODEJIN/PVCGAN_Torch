from yin import pitch_calc

import librosa, yaml, os
import crepe
import numpy as np
import matplotlib.pyplot as plt 
from scipy.ndimage import gaussian_filter1d

with open('Hyper_Parameter.yaml') as f:
    hp_Dict = yaml.load(f, Loader=yaml.Loader)

os.environ['CUDA_VISIBLE_DEVICES']= '1'

sig,  sr = librosa.load('D:/Pattern/Sing/NUS48E/ADIZ/sing/01.wav')
sig = librosa.util.normalize(sig)

pitch = pitch_calc(
    sig= sig,
    sr= sr,
    w_len= 1024,
    w_step= 256,
    confidence_threshold= hp_Dict['Sound']['Confidence_Threshold'],
    gaussian_smoothing_sigma= hp_Dict['Sound']['Gaussian_Smoothing_Sigma']
    )

# pitch_yin = pitch

# pitch, confidence = crepe.predict(
#         audio= sig,
#         sr= sr,
#         viterbi=True
#         )[1:3]

# confidence = np.where(confidence > hp_Dict['Sound']['Confidence_Threshold'], confidence, 0)
# pitch = np.where(confidence > 0, pitch, 0)
# pitch = gaussian_filter1d(pitch, sigma= hp_Dict['Sound']['Gaussian_Smoothing_Sigma'])
# pitch /= np.max(pitch) + 1e-7
# pitch_crepe = pitch

# plt.subplot(311)
# plt.plot(sig)
# plt.subplot(312)
# plt.plot(pitch_yin)
# plt.subplot(313)
# plt.plot(pitch_crepe)
# plt.show()


import torch
from Modules import LinearUpsample1D

pitch = np.expand_dims(pitch[500:600], axis= 0)    # [Batch, time]
pitch_up = LinearUpsample1D(256)(torch.tensor(pitch)).numpy()

plt.subplot(211)
plt.plot(pitch[0])
plt.subplot(212)
plt.plot(pitch_up[0])
plt.show()
