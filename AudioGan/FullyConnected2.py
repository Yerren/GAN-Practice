import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import os
import datetime
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import tensorflow as tf
from tensorflow.keras import layers

# We never take the last chunk of the track, as it is most likely to have blank audio.
# (maybe try without first chunk too, as it often has variation)
REAL_FILE_SIZE = 5314197

CHUNK_SIZE = 4428  # Must be even
FILE_SIZE = (5314197 // CHUNK_SIZE) * CHUNK_SIZE


CHUNKS_PER_FILE = FILE_SIZE // CHUNK_SIZE

# NUM_TRACKS = 1438
NUM_TRACKS = 1438


config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
# config.gpu_options.per_process_gpu_memory_fraction = 0.7
sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)

# # To load data from npz file:
# X = np.load("datasets/Dataset0.npy")
# # print(X)
# # print(np.sum(~X.any(1)))
# np.random.shuffle(X)
# # print(X)
# print(np.sum(~X.any(1)))
# X = np.expand_dims(X, axis=2)



# NUM_TRACKS = 50000
filenames = np.array(["track ({}).wav".format(i) for i in range(1, NUM_TRACKS+1)])
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

def create_array(array_in):
    result_array_real = array_in.real
    result_array_comp = array_in.imag

    final_array = np.concatenate([result_array_real, result_array_comp])
    return final_array


class My_Custom_Generator_v1(tf.keras.utils.Sequence):

    def __init__(self, image_filenames, labels, batch_size):
        # "labels" not actually used, as input data = output data
        self.image_filenames = image_filenames
        self.labels = labels
        self.batch_size = batch_size

    def __len__(self):
        return (np.ceil(len(self.image_filenames) / float(self.batch_size))).astype(np.int)

    def __getitem__(self, idx):
        batch_x = self.image_filenames[idx * self.batch_size: (idx + 1) * self.batch_size]
        result_array = np.array([create_array(file_name) for file_name in batch_x])
        return result_array, result_array


class My_Custom_Generator(tf.keras.utils.Sequence):

    def __init__(self, image_filenames, labels, batch_size):
        # "labels" not actually used, as input data = output data
        self.image_filenames = image_filenames
        self.labels = labels
        self.batch_size = batch_size

    def __len__(self):
        return (np.ceil((len(self.image_filenames) * CHUNKS_PER_FILE) / float(self.batch_size))).astype(np.int)

    def __getitem__(self, idx):
        batches_item_length = self.batch_size * CHUNK_SIZE

        index = batches_item_length * idx

        file_num = index // FILE_SIZE
        position = index % FILE_SIZE

        rate, data = wavfile.read("audio/wav/" + self.image_filenames[file_num])
        data = data - np.mean(data)

        batch_result = []

        test_array = []
        for i in range(self.batch_size):

            if (position + 1)*CHUNK_SIZE <= data.shape[0]:
                track = data[position*CHUNK_SIZE:(position + 1) * CHUNK_SIZE]
                test_array += track.tolist()
                dataFft = np.fft.rfft(track)
                dataFft = dataFft / np.absolute(dataFft).max()
                dataFft[np.abs(dataFft) < 0.001] = 0
                if not np.all(dataFft == 0) and not np.isnan(dataFft).any() and not np.isinf(dataFft).any():
                    batch_result.append(create_array(dataFft))
                else:
                    raise RuntimeError('Bad values after Fft.')

                position += 1
            else:
                file_num += 1
                file_num = file_num % len(self.image_filenames)
                position = 0

        test_array = np.array(test_array)

        result_array = np.array(batch_result)

        return result_array, result_array

def make_aa_model():
    size = CHUNK_SIZE + 2
    model = tf.keras.Sequential()

    model.add(layers.Dense(1800, activation='relu', input_shape=[size]))
    model.add(layers.Dense(1500, activation='relu'))
    model.add(layers.Dense(1200, activation='relu'))
    model.add(layers.Dense(900, activation='linear'))
    model.add(layers.Dense(1200, activation='relu'))
    model.add(layers.Dense(1500, activation='relu'))
    model.add(layers.Dense(1800, activation='relu'))
    model.add(layers.Dense(size, activation='linear'))

    assert model.output_shape == (None, size)

    return model


def write_audio(index_in, postfix, rate=44100):
    one_song_gen = My_Custom_Generator(filenames, filenames, CHUNKS_PER_FILE)
    input_matrix, _ = one_song_gen.__getitem__(index_in)

    result_matrix = aa_model.predict(input_matrix)
    # result_matrix = input_matrix

    output_song = np.array([])
    for row in result_matrix:
        array_real = row[:row.shape[0]//2]
        array_comp = row[row.shape[0]//2:]

        joined_array = array_real + 1j*array_comp

        dataInvFft = np.fft.irfft(joined_array)
        dataInvFft = dataInvFft / dataInvFft.max()
        output_song = np.concatenate((output_song, dataInvFft))

    wavfile.write("out/out_{}.wav".format(postfix), rate, output_song)

aa_model = make_aa_model()

optimizer = tf.keras.optimizers.Adam(0.00001)

aa_model.compile(loss='mean_squared_error',
        optimizer=optimizer,
        metrics=['mean_absolute_error', 'mean_squared_error'])


aa_model.summary()


checkpoint_path = "training/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(
    checkpoint_path, verbose=1, save_weights_only=True,
    # Save weights, every 5-epochs.
    period=50)

class SaveTrack(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 5 == 0:
        write_audio(0, "trPred_{}".format(epoch))
        write_audio(NUM_TRACKS - 1, "valPred_{}".format(epoch))


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

BATCH_SIZE = 64

train_generator = My_Custom_Generator(filenames_tr, filenames_tr, BATCH_SIZE)
validation_generator = My_Custom_Generator(filenames_te, filenames_te, BATCH_SIZE)

# latest = tf.train.latest_checkpoint("checkpoints")
# print(checkpoint_dir)
# aa_model.load_weights(latest)

log_dir = os.path.join(
    "logs",
    "fit",
    datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
)
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, mode='min')

write_audio(0, "trPred_{}".format(0))


history = aa_model.fit_generator(
  generator=train_generator, validation_data=validation_generator, epochs=500, verbose=1,
    callbacks=[SaveTrack(), cp_callback, tensorboard_callback, early_stop])


aa_model.save_weights('./checkpoints/final_checkpoint')
plot_history([(" ", history)])
