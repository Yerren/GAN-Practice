import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import os
import tensorflow as tf
from tensorflow.keras import layers

# We never take the last chunk of the track, as it is most likely to have blank audio.
# (maybe try without first chunk too, as it often has variation)

NUM_FILES = 50
FILE_LENGTH = 5314197  # We assume preset length (2 minutes)
CHUNK_SIZE = 44284

# How many chunks to ignore from each end of the file
INIT_CUTOFF = 5
END_CUTOFF = 5

ROWS_PER_FILE = (FILE_LENGTH//CHUNK_SIZE - INIT_CUTOFF - END_CUTOFF)

# # To save data as npz files:
# for BATCH_NO in range(1):
#     X = np.empty((NUM_FILES * ROWS_PER_FILE, CHUNK_SIZE)).astype(np.float32)
#     row_num = 0
#     for i in range(1, NUM_FILES+1):
#         mywav = "audio/wav/track ({}).wav".format(i + BATCH_NO*NUM_FILES)
#         # print(mywav)
#         rate, data = wavfile.read(mywav)
#         data = data / data.max()
#
#         inPos = INIT_CUTOFF
#         # Should probably do some safety checks here
#
#         while (inPos+END_CUTOFF)*CHUNK_SIZE < data.shape[0]:
#             X[row_num] = data[inPos*CHUNK_SIZE:(inPos + 1) * CHUNK_SIZE]
#             print(X[row_num])
#             print(row_num)
#             inPos +=1
#             row_num += 1
#
#     X = X[~np.all(X == 0, axis=1)]
#     np.save("datasets/Dataset{}.npy".format(BATCH_NO), X.astype(np.float32))
#     del X
#
#

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
# config.gpu_options.per_process_gpu_memory_fraction = 0.7
sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)


NUM_TRACKS = 40000
filenames = np.array(["track {}.npy".format(i) for i in range(NUM_TRACKS)])
np.random.shuffle(filenames)

filenames_tr = filenames[:int(NUM_TRACKS * 0.8)]
filenames_te = filenames[int(NUM_TRACKS * 0.8):]

# V1:
# def make_aa_model_v1():
#     model = tf.keras.Sequential()
#     # input = 1 x 44284
#     model.add(layers.Conv1D(100, 100, activation='relu', padding="same", input_shape=(44284, 1)))  # output = 100 x 442849
#     model.add(layers.MaxPooling1D(5, padding="same"))  # output = 100 x 8857 (44284 / 5)
#     model.add(layers.Conv1D(50, 56, activation='relu', padding="same"))  # output = 50 x 8857
#     model.add(layers.MaxPooling1D(5, padding="same"))  # output = 50 x 1772 (8857/ 5)
#     model.add(layers.Conv1D(50, 56, activation='relu', padding="same"))  # output = 50 x 1772
#     model.add(layers.MaxPooling1D(5, padding="same"))  # output = 50 x 355 (1772/ 5)
#
#     assert model.output_shape == (None, 355, 50)
#     model.add(layers.Conv1D(5, 100, activation='relu', padding="same"))  # output = 50 x 356
#     model.add(layers.UpSampling1D(5))  # output = 50 x 1775
#     model.add(layers.Conv1D(50, 100, activation='relu', padding="same"))  # output = 50 x 1775
#     model.add(layers.UpSampling1D(5))  # output = 50 x 8875
#     model.add(layers.Conv1D(50, 100, activation='relu', padding="same"))  # output = 50 x 8875
#     model.add(layers.UpSampling1D(5))  # output = 50 x 44375
#     model.add(layers.Conv1D(1, 92, activation='relu', padding="valid"))  # output = 1 x 44284 (using VALID)
#
#     assert model.output_shape == (None, 44284, 1)
#     print("C")
#
#     return model

def make_aa_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv1D(100, 10, activation='relu', padding="same", input_shape=(22143, 1)))
    model.add(layers.MaxPooling1D(5, padding="same"))  # output = 100 x 4429
    model.add(layers.Conv1D(50, 10, activation='relu', padding="same"))  # output = 50 x 8857
    model.add(layers.MaxPooling1D(5, padding="same"))  # output = 50 x 886 (8857/ 5)

    assert model.output_shape == (None, 886, 50)
    model.add(layers.Conv1D(50, 10, activation='relu', padding="same"))  # output = 50 x 886
    model.add(layers.UpSampling1D(5))  # output = 50 x 4430
    model.add(layers.Conv1D(50, 10, activation='relu', padding="same"))  # output = 50 x 4430
    model.add(layers.UpSampling1D(5))  # output = 50 x 44300
    model.add(layers.Conv1D(1, 8, activation='relu', padding="valid"))  # output = 1 x 22150 (using VALID)

    assert model.output_shape == (None, 22143, 1)
    print("C")

    return model

aa_model = make_aa_model()

optimizer = tf.keras.optimizers.Adam(0.0005)

aa_model.compile(loss='mean_squared_error',
        optimizer=optimizer,
        metrics=['mean_absolute_error', 'mean_squared_error'])

checkpoint_path = "training/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(
    checkpoint_path, verbose=1, save_weights_only=True,
    # Save weights, every 5-epochs.
    period=5)

def write_audio(array_in, postfix, rate=44100):
    dataInvFft = np.fft.irfft(array_in)
    dataInvFft = dataInvFft / dataInvFft.max()
    wavfile.write("out_{}.wav".format(postfix), rate, dataInvFft)


trSave = np.load('datasets/IndividualTracks/track 10.npy')
teSave = np.load('datasets/IndividualTracks/track 172000.npy')




class SaveTrack(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 5 == 0:
        result = aa_model.predict(np.array([trSave]))
        write_audio(result[0].flatten(), "trPred_{}".format(epoch))
        write_audio(trSave.flatten(), "trAct_{}".format(epoch))
        result = aa_model.predict(np.array([teSave]))
        write_audio(result[0].flatten(), "tePred_{}".format(epoch))
        write_audio(teSave.flatten(), "teAct_{}".format(epoch))


def plot_history(histories, key='mean_absolute_error'):
  plt.figure(figsize=(16,10))

  for name, history in histories:
    val = plt.plot(history.epoch, history.history['val_'+key],
                   '--', label=name.title()+' Val')
    plt.plot(history.epoch, history.history[key], color=val[0].get_color(),
             label=name.title()+' Train')

  plt.xlabel('Epochs')
  plt.ylabel(key.replace('_',' ').title())
  plt.legend()

  plt.xlim([0, max(history.epoch)])
  plt.show()


# V1:
# def make_aa_model_v1():
#     model = tf.keras.Sequential()
#     # input = 1 x 44284
#     model.add(layers.Conv1D(100, 100, activation='relu', padding="same", input_shape=(44284, 1)))  # output = 100 x 442849
#     model.add(layers.MaxPooling1D(5, padding="same"))  # output = 100 x 8857 (44284 / 5)
#     model.add(layers.Conv1D(50, 56, activation='relu', padding="same"))  # output = 50 x 8857
#     model.add(layers.MaxPooling1D(5, padding="same"))  # output = 50 x 1772 (8857/ 5)
#     model.add(layers.Conv1D(50, 56, activation='relu', padding="same"))  # output = 50 x 1772
#     model.add(layers.MaxPooling1D(5, padding="same"))  # output = 50 x 355 (1772/ 5)
#
#     assert model.output_shape == (None, 355, 50)
#     model.add(layers.Conv1D(5, 100, activation='relu', padding="same"))  # output = 50 x 356
#     model.add(layers.UpSampling1D(5))  # output = 50 x 1775
#     model.add(layers.Conv1D(50, 100, activation='relu', padding="same"))  # output = 50 x 1775
#     model.add(layers.UpSampling1D(5))  # output = 50 x 8875
#     model.add(layers.Conv1D(50, 100, activation='relu', padding="same"))  # output = 50 x 8875
#     model.add(layers.UpSampling1D(5))  # output = 50 x 44375
#     model.add(layers.Conv1D(1, 92, activation='relu', padding="valid"))  # output = 1 x 44284 (using VALID)
#
#     assert model.output_shape == (None, 44284, 1)
#     print("C")
#
#     return model

# def make_aa_model_v2():
#     model = tf.keras.Sequential()
#     # input = 1 x 44284
#     model.add(layers.Conv1D(100, 10, activation='relu', padding="same", input_shape=(44284, 1)))  # output = 100 x 442849
#     model.add(layers.MaxPooling1D(5, padding="same"))  # output = 100 x 8857 (44284 / 5)
#     model.add(layers.Conv1D(50, 10, activation='relu', padding="same"))  # output = 50 x 8857
#     model.add(layers.MaxPooling1D(5, padding="same"))  # output = 50 x 1772 (8857/ 5)
#
#     assert model.output_shape == (None, 1772, 50)
#     model.add(layers.Conv1D(50, 10, activation='relu', padding="same"))  # output = 50 x 1772
#     model.add(layers.UpSampling1D(5))  # output = 50 x 8860
#     model.add(layers.Conv1D(50, 10, activation='relu', padding="same"))  # output = 50 x 8875
#     model.add(layers.UpSampling1D(5))  # output = 50 x 44300
#     model.add(layers.Conv1D(1, 17, activation='relu', padding="valid"))  # output = 1 x 44284 (using VALID)
#
#     assert model.output_shape == (None, 44284, 1)
#     print("C")
#
#     return model

# Credit to: https://medium.com/@mrgarg.rajat/training-on-large-datasets-that-dont-fit-in-memory-in-keras-60a974785d71
# for the bulk of the generator
class My_Custom_Generator(tf.keras.utils.Sequence):

    def __init__(self, image_filenames, labels, batch_size):
        # "labels" not actually used, as input data = output data
        self.image_filenames = image_filenames
        self.labels = labels
        self.batch_size = batch_size

    def __len__(self):
        return (np.ceil(len(self.image_filenames) / float(self.batch_size))).astype(np.int)

    def __getitem__(self, idx):
        batch_x = self.image_filenames[idx * self.batch_size: (idx + 1) * self.batch_size]

        result_array = np.array([np.load('datasets/IndividualTracks/' + str(file_name)) for file_name in batch_x])
        result_array = np.expand_dims(result_array, axis=2)
        return result_array, result_array


BATCH_SIZE = 32

train_generator = My_Custom_Generator(filenames_tr, filenames_tr, BATCH_SIZE)
validation_generator = My_Custom_Generator(filenames_te, filenames_te, BATCH_SIZE)

history = aa_model.fit_generator(
  generator=train_generator, validation_data=validation_generator, epochs=500, verbose=1,
    callbacks=[SaveTrack(), cp_callback])


aa_model.save_weights('./checkpoints/final_checkpoint')
plot_history([(" ", history)])
