# from yin import pitch_calc

# import librosa, yaml, os
# import crepe
# import numpy as np
# import matplotlib.pyplot as plt 
# from scipy.ndimage import gaussian_filter1d

# with open('Hyper_Parameter.yaml') as f:
#     hp_Dict = yaml.load(f, Loader=yaml.Loader)

# os.environ['CUDA_VISIBLE_DEVICES']= '1'

# sig,  sr = librosa.load('D:/Pattern/Sing/NUS48E/ADIZ/sing/01.wav')
# sig = librosa.util.normalize(sig)

# pitch = pitch_calc(
#     sig= sig,
#     sr= sr,
#     w_len= 1024,
#     w_step= 256,
#     confidence_threshold= hp_Dict['Sound']['Confidence_Threshold'],
#     gaussian_smoothing_sigma= hp_Dict['Sound']['Gaussian_Smoothing_Sigma']
#     )

# # pitch_yin = pitch

# # pitch, confidence = crepe.predict(
# #         audio= sig,
# #         sr= sr,
# #         viterbi=True
# #         )[1:3]

# # confidence = np.where(confidence > hp_Dict['Sound']['Confidence_Threshold'], confidence, 0)
# # pitch = np.where(confidence > 0, pitch, 0)
# # pitch = gaussian_filter1d(pitch, sigma= hp_Dict['Sound']['Gaussian_Smoothing_Sigma'])
# # pitch /= np.max(pitch) + 1e-7
# # pitch_crepe = pitch

# # plt.subplot(311)
# # plt.plot(sig)
# # plt.subplot(312)
# # plt.plot(pitch_yin)
# # plt.subplot(313)
# # plt.plot(pitch_crepe)
# # plt.show()


# import torch
# from Modules import LinearUpsample1D

# pitch = np.expand_dims(pitch[500:600], axis= 0)    # [Batch, time]
# pitch_up = LinearUpsample1D(256)(torch.tensor(pitch)).numpy()

# plt.subplot(211)
# plt.plot(pitch[0])
# plt.subplot(212)
# plt.plot(pitch_up[0])
# plt.show()


# from scipy.io import wavfile 
# import pickle
# import numpy as np

# with open('NUS48E.JLEE05.PICKLE', 'rb') as f:
#     x = pickle.load(f)

# wavfile.write(
#     filename='D:/x.wav',
#     data= (x['Signal'] * 32767.5).astype(np.int16),
#     rate= 16000
#     )

# singers = [
#     'ADIZ',
#     'JLEE',
#     'JTAN',
#     'KENN',
#     'MCUR',
#     'MPOL',
#     'MPUR',
#     'NJAT',
#     'PMAR',
#     'SAMF',
#     'VKOW',
#     'ZHIY'
#     ]

# paths = [
#     'C:/Pattern/PN.Pattern.NUS48E/NUS48E.ADIZ01.PICKLE',
#     'C:/Pattern/PN.Pattern.NUS48E/NUS49E.JLEE05.PICKLE',
#     'C:/Pattern/PN.Pattern.NUS48E/NUS50E.JTAN07.PICKLE',
#     'C:/Pattern/PN.Pattern.NUS48E/NUS51E.KENN04.PICKLE',
#     'C:/Pattern/PN.Pattern.NUS48E/NUS52E.MCUR10.PICKLE',
#     'C:/Pattern/PN.Pattern.NUS48E/NUS53E.MPOL11.PICKLE',
#     'C:/Pattern/PN.Pattern.NUS48E/NUS54E.MPUR02.PICKLE',
#     'C:/Pattern/PN.Pattern.NUS48E/NUS55E.NJAT15.PICKLE',
#     'C:/Pattern/PN.Pattern.NUS48E/NUS56E.PMAR08.PICKLE',
#     'C:/Pattern/PN.Pattern.NUS48E/NUS57E.SAMF09.PICKLE',
#     'C:/Pattern/PN.Pattern.NUS48E/NUS58E.VKOW19.PICKLE',
#     'C:/Pattern/PN.Pattern.NUS48E/NUS59E.ZHIY03.PICKLE'
#     ]

# songs = [
#     '01',
#     '05',
#     '07',
#     '04',
#     '10',
#     '11',
#     '02',
#     '15',
#     '08',
#     '09',
#     '19',
#     '03',
#     ]

# exports = ['Source_Label\tSource_Path\tConversion_Label\tConversion_Singer\tStart_Index\tEnd_Index']
# for source_Singer, source_Song, path in zip(singers, songs, paths):
#     for index, conversion_Singer in enumerate(singers):
#         exports.append('{}_{}\t{}\t{}\t{}\t3000\t4280'.format(source_Singer, source_Song, path, conversion_Singer, index))

# open('Inference_for_Training.txt', 'w').write('\n'.join(exports))


import librosa
from scipy.io import wavfile
import os
import numpy as np


paths = [
    'D:/Python_Programming/CODEJIN.github.io/PVCGAN/ref/original/ADIZ_01.wav',
    'D:/Python_Programming/CODEJIN.github.io/PVCGAN/ref/original/JLEE_05.wav',
    'D:/Python_Programming/CODEJIN.github.io/PVCGAN/ref/original/JTAN_07.wav',
    'D:/Python_Programming/CODEJIN.github.io/PVCGAN/ref/original/KENN_04.wav',
    'D:/Python_Programming/CODEJIN.github.io/PVCGAN/ref/original/MCUR_10.wav',
    'D:/Python_Programming/CODEJIN.github.io/PVCGAN/ref/original/MPOL_11.wav',
    'D:/Python_Programming/CODEJIN.github.io/PVCGAN/ref/original/MPUR_02.wav',
    'D:/Python_Programming/CODEJIN.github.io/PVCGAN/ref/original/NJAT_15.wav',
    'D:/Python_Programming/CODEJIN.github.io/PVCGAN/ref/original/PMAR_08.wav',
    'D:/Python_Programming/CODEJIN.github.io/PVCGAN/ref/original/SAMF_09.wav',
    'D:/Python_Programming/CODEJIN.github.io/PVCGAN/ref/original/VKOW_19.wav',
    'D:/Python_Programming/CODEJIN.github.io/PVCGAN/ref/original/ZHIY_03.wav',
    ]

for path in paths:
    audio = librosa.core.load(path, sr= 24000)[0]
    audio = librosa.effects.trim(audio, top_db=20, frame_length= 512, hop_length= 256)[0]
    audio = librosa.util.normalize(audio)
    audio = audio[3000*256:4280*256]
    wavfile.write(
        filename= os.path.basename(path),
        data= (np.clip(audio, -1.0 + 1e-7, 1.0 - 1e-7) * 32767.5).astype(np.int16),
        rate= 24000
        )
    
