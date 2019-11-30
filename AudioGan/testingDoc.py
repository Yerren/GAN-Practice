import numpy as np
from scipy.io import wavfile

# FACTOR = 10
#
# mywav = "audio/wav/track (1).wav"
#
# rate, data = wavfile.read(mywav)
# print(data)
# print(data.shape[0])
# dataReduced = data[range(0, data.shape[0], FACTOR)]
#
# rateReduced = rate//FACTOR
#
# print(dataReduced.shape[0])
# print(dataReduced)
# print(rateReduced)

# mywav = "audio/wav/track (1).wav"
#
#
# rate, data = wavfile.read(mywav)
# data = data - np.mean(data)
#
# dataFft = np.fft.rfft(data)
# dataFft = dataFft / dataFft.max()
# print(dataFft)
# dataFft[np.abs(dataFft) < 0.0001] = 0
# print(len(np.where(np.abs(dataFft) < 0.0001)[0]))
#
# dataInvFft = np.fft.irfft(dataFft)
# dataInvFft = dataInvFft / dataInvFft.max()
# print(dataInvFft)
#
# wavfile.write("out1.wav", rate, dataInvFft)


# for i in range(172792):
#     # print(i)
#     dataFft = np.load("datasets/IndividualTracks/track {}.npy".format(i))
#     if dataFft.shape != (22143,):
#         print(i)
#     dataInvFft = np.fft.irfft(dataFft)
#     dataInvFft = dataInvFft / dataInvFft.max()
#
#     # wavfile.write("out1.wav", 44100, dataInvFft)





