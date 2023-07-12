import tensorflow as tf
from tensorflow import keras


class Spectrogram(keras.layers.Layer):
    def __init__(self, n_dft=256, n_hop=128, **kwargs):
        super(Spectrogram, self).__init__(**kwargs)
        self.n_dft = n_dft
        self.n_hop = n_hop

    def call(self, inputs):
        spectrogram = tf.signal.stft(inputs, frame_length=self.n_dft, frame_step=self.n_hop)
        spectrogram = tf.abs(spectrogram)
        return spectrogram

    def get_config(self):
        config = super(Spectrogram, self).get_config()
        config.update({'frame_length': self.n_dft, 'frame_step': self.n_hop})
        return config

