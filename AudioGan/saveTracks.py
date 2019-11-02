import numpy as np
from scipy.io import wavfile

# NUM_FILES = 1438
NUM_FILES = 2
FILE_LENGTH = 5314197  # We assume preset length (2 minutes)
CHUNK_SIZE = 44284

# How many chunks to ignore from each end of the file
INIT_CUTOFF = 0
END_CUTOFF = 0

ROWS_PER_FILE = (FILE_LENGTH//CHUNK_SIZE - INIT_CUTOFF - END_CUTOFF)

# To save data as npz files:
row_num = 0
for i in range(NUM_FILES):
    mywav = "audio/wav/track ({}).wav".format(i + 1)

    rate, data = wavfile.read(mywav)
    data = data - np.mean(data)

    dataFft = np.fft.rfft(data)
    dataFft = dataFft / np.absolute(dataFft).max()
    dataFft[np.abs(dataFft) < 0.001] = 0

    # print(dataFft.min())

#     if not np.all(dataFft == 0) and not np.isnan(dataFft).any() and not np.isinf(dataFft).any():
#         np.save("datasets/IndividualTracks2/track {}.npy".format(row_num), dataFft)
#         print("Finished item {}".format(row_num))
#         row_num += 1
#     else:
#         print("Bad Track")
#
# dataFft2 = np.load("datasets/IndividualTracks2/track 1.npy")
# # print(dataFft)
# dataInvFft = np.fft.irfft(dataFft2)
# dataInvFft = dataInvFft / dataInvFft.max()
#
# wavfile.write("out1.wav", 44100, dataInvFft)

# row_num = 0
# for i in range(NUM_FILES):
#     mywav = "audio/wav/track ({}).wav".format(i + 1)
#
#     rate, data = wavfile.read(mywav)
#     data = data - np.mean(data)
#
#     inPos = INIT_CUTOFF
#     while (inPos+END_CUTOFF+1)*CHUNK_SIZE < data.shape[0]:
#         track = data[inPos*CHUNK_SIZE:(inPos + 1) * CHUNK_SIZE]
#
#         dataFft = np.fft.rfft(track)
#         dataFft = dataFft / np.absolute(dataFft).max()
#         dataFft[np.abs(dataFft) < 0.001] = 0
#
#         if not np.all(dataFft == 0) and not np.isnan(dataFft).any() and not np.isinf(dataFft).any():
#             np.save("datasets/IndividualTracks/track {}.npy".format(row_num), dataFft)
#             print("Finished item {}".format(row_num))
#             row_num += 1
#         else:
#             print("Bad Track")
#
#         inPos += 1

# dataFft = np.load("datasets/IndividualTracks/track 1.npy")
# # print(dataFft)
# dataInvFft = np.fft.irfft(dataFft)
# dataInvFft = dataInvFft / dataInvFft.max()
#
# wavfile.write("out1.wav", 44100, dataInvFft)
