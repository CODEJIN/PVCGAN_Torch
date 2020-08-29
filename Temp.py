# from yin import pitch_calc

# import librosa, yaml, os
# import crepe
# import numpy as np
# import matplotlib.pyplot as plt 
# from scipy.ndimage import gaussian_filter1d

# with open('Hyper_Parameters.yaml') as f:
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
#     'C:/Pattern/PN.Pattern.NUS48E_Pitch_Testing/NUS48E.ADIZ01.PICKLE',
#     'C:/Pattern/PN.Pattern.NUS48E_Pitch_Testing/NUS49E.JLEE05.PICKLE',
#     'C:/Pattern/PN.Pattern.NUS48E_Pitch_Testing/NUS50E.JTAN07.PICKLE',
#     'C:/Pattern/PN.Pattern.NUS48E_Pitch_Testing/NUS51E.KENN04.PICKLE',
#     'C:/Pattern/PN.Pattern.NUS48E_Pitch_Testing/NUS52E.MCUR10.PICKLE',
#     'C:/Pattern/PN.Pattern.NUS48E_Pitch_Testing/NUS53E.MPOL11.PICKLE',
#     'C:/Pattern/PN.Pattern.NUS48E_Pitch_Testing/NUS54E.MPUR02.PICKLE',
#     'C:/Pattern/PN.Pattern.NUS48E_Pitch_Testing/NUS55E.NJAT15.PICKLE',
#     'C:/Pattern/PN.Pattern.NUS48E_Pitch_Testing/NUS56E.PMAR08.PICKLE',
#     'C:/Pattern/PN.Pattern.NUS48E_Pitch_Testing/NUS57E.SAMF09.PICKLE',
#     'C:/Pattern/PN.Pattern.NUS48E_Pitch_Testing/NUS58E.VKOW19.PICKLE',
#     'C:/Pattern/PN.Pattern.NUS48E_Pitch_Testing/NUS59E.ZHIY03.PICKLE'
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

# exports = ['Source_Label/tSource_Path/tConversion_Label/tConversion_Singer/tStart_Index/tEnd_Index']
# for source_Singer, source_Song, path in zip(singers, songs, paths):
#     for index, conversion_Singer in enumerate(singers):
#         exports.append('{}_{}/t{}/t{}/t{}/t3000/t4280'.format(source_Singer, source_Song, path, conversion_Singer, index))

# open('Inference_for_Training.txt', 'w').write('/n'.join(exports))


# import librosa
# from scipy.io import wavfile
# import os
# import numpy as np


# paths = [
#     'D:/Python_Programming/CODEJIN.github.io/PVCGAN/ref/original/ADIZ_01.wav',
#     'D:/Python_Programming/CODEJIN.github.io/PVCGAN/ref/original/JLEE_05.wav',
#     'D:/Python_Programming/CODEJIN.github.io/PVCGAN/ref/original/JTAN_07.wav',
#     'D:/Python_Programming/CODEJIN.github.io/PVCGAN/ref/original/KENN_04.wav',
#     'D:/Python_Programming/CODEJIN.github.io/PVCGAN/ref/original/MCUR_10.wav',
#     'D:/Python_Programming/CODEJIN.github.io/PVCGAN/ref/original/MPOL_11.wav',
#     'D:/Python_Programming/CODEJIN.github.io/PVCGAN/ref/original/MPUR_02.wav',
#     'D:/Python_Programming/CODEJIN.github.io/PVCGAN/ref/original/NJAT_15.wav',
#     'D:/Python_Programming/CODEJIN.github.io/PVCGAN/ref/original/PMAR_08.wav',
#     'D:/Python_Programming/CODEJIN.github.io/PVCGAN/ref/original/SAMF_09.wav',
#     'D:/Python_Programming/CODEJIN.github.io/PVCGAN/ref/original/VKOW_19.wav',
#     'D:/Python_Programming/CODEJIN.github.io/PVCGAN/ref/original/ZHIY_03.wav',
#     ]

# for path in paths:
#     audio = librosa.core.load(path, sr= 24000)[0]
#     audio = librosa.effects.trim(audio, top_db=20, frame_length= 512, hop_length= 256)[0]
#     audio = librosa.util.normalize(audio)
#     audio = audio[3000*256:4280*256]
#     wavfile.write(
#         filename= os.path.basename(path),
#         data= (np.clip(audio, -1.0 + 1e-7, 1.0 - 1e-7) * 32767.5).astype(np.int16),
#         rate= 24000
#         )
    
# # from Modules import PVCGAN
# from Logger import Logger

# import os, torch, yaml

# os.environ['CUDA_VISIBLE_DEVICES']= '-1'

# logger = Logger('/home/heejo/Documents/Personal/T')

# with open('Hyper_Parameters.yaml') as f:
#     hp_Dict = yaml.load(f, Loader=yaml.Loader)
# logger.add_hparams({'a': 10, 'b': 0.2315412}, {}, global_step= 0)


# import pickle
# import matplotlib.pyplot as plt

# jlee = pickle.load(open("C:/Pattern/PN.Pattern.NUS48E_Pitch_Testing/NUS48E.JLEE11.PICKLE", 'rb'))
# mpol = pickle.load(open("C:/Pattern/PN.Pattern.NUS48E_Pitch_Testing/NUS48E.MPOL11.PICKLE", 'rb'))
# pmar = pickle.load(open("C:/Pattern/PN.Pattern.NUS48E_Pitch_Testing/NUS48E.PMAR11.PICKLE", 'rb'))
# vkow = pickle.load(open("C:/Pattern/PN.Pattern.NUS48E_Pitch_Testing/NUS48E.VKOW11.PICKLE", 'rb'))

# plt.subplot(411)
# plt.plot(jlee['Pitch'])
# plt.subplot(412)
# plt.plot(mpol['Pitch'])
# plt.subplot(413)
# plt.plot(pmar['Pitch'])
# plt.subplot(414)
# plt.plot(vkow['Pitch'])
# plt.tight_layout()
# plt.show()


# import os

# for root, _, files in os.walk('C:/Pattern/PN.Pattern.NUS48E_Pitch_Testing'):
#     for file in files:
#         file = os.path.join(root, file)
#         x = pickle.load(open(file, 'rb'))
#         if 'Pitch' in x.keys():
#             print(file, min(x['Pitch']), max(x['Pitch']))






# Referece: https://github.com/janfreyberg/pytorch-revgrad

# import torch

# class Func(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, input_, x):
#         ctx.save_for_backward(input_)
#         ctx.x = x
#         output = input_
#         return output

#     @staticmethod
#     def backward(ctx, grad_output):  # pragma: no cover
#         grad_input = None
#         if ctx.needs_input_grad[0]:
#             grad_input = -grad_output * ctx.x
#         return grad_input, None

# class GRL(torch.nn.Module):
#     def __init__(self, weight):
#         """
#         A gradient reversal layer.
#         This layer has no parameters, and simply reverses the gradient
#         in the backward pass.
#         """
#         super(GRL, self).__init__()
#         self.weight = weight

#     def forward(self, input_):
#         return Func.apply(input_, torch.FloatTensor([self.weight]))


# a = torch.nn.Conv1d(4, 5, 3, 1, 1)
# b = GRL(.5)
# c = torch.nn.Conv1d(5, 4, 3, 1, 1)

# optim = torch.optim.SGD(
#     list(a.parameters()) + list(b.parameters()) + list(c.parameters()),
#     0.1
#     )

# x = torch.randn(1, 4, 1)

# y = c(a(x))
# loss = torch.nn.MSELoss()(x, y)
# optim.zero_grad()
# loss.backward()
# print(a.weight.grad)
# print('#' * 100)

# y = c(b(a(x)))
# loss = torch.nn.MSELoss()(x, y)
# optim.zero_grad()
# loss.backward()
# print(a.weight.grad)







# from yin import pitch_calc

# import librosa, yaml, os
# import numpy as np
# import matplotlib.pyplot as plt 
# from Audio import Audio_Prep

# audio1 = Audio_Prep('D:/Pattern/Sing/NUS48E/PMAR/sing/05.wav', 24000, 20)
# pitch1 = pitch_calc(
#     sig= audio1,
#     sr= 24000,
#     w_len= 1024,
#     w_step= 256,
#     confidence_threshold= 0.6,
#     gaussian_smoothing_sigma= 0.0,
#     f0_min=100,
#     f0_max=500,
#     )
# audio2 = Audio_Prep('D:/Pattern/Sing/NUS48E/JLEE/sing/05.wav', 24000, 20)
# pitch2 = pitch_calc(
#     sig= audio2,
#     sr= 24000,
#     w_len= 1024,
#     w_step= 256,
#     confidence_threshold= 0.6,
#     gaussian_smoothing_sigma= 0.0,
#     f0_min=100,
#     f0_max=500,
#     )

# print(min(pitch1))
# print(min(pitch2))

# print(np.mean(pitch1))
# print(np.mean(pitch2))

# plt.subplot(221)
# plt.plot(audio1)
# plt.subplot(222)
# plt.plot(audio2)
# plt.subplot(223)
# plt.plot(pitch1)
# plt.subplot(224)
# plt.plot(pitch2)
# plt.tight_layout()
# plt.show()

import pickle
import numpy as np

print(np.mean(pickle.load(open("C:/Pattern/PN.Pattern.NUS48E/NUS48E.PMAR05.PICKLE", 'rb'))['Pitch']))
print(np.mean(pickle.load(open("C:/Pattern/PN.Pattern.NUS48E/NUS48E.JLEE05.PICKLE", 'rb'))['Pitch']))