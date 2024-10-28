import numpy as np
import tensorflow as tf


class NonLocalBlock(tf.keras.layers.Layer):
    def __init__(self, units=1, reduction = 1):


        super(NonLocalBlock, self).__init__()
        self.units = units #number of filters of previous convolutional block (or number of channels)
        self.internal_units = max(units // 2, 1)



        self.x = tf.keras.layers.Conv1D(filters = self.internal_units, kernel_size= 1, padding='same', kernel_initializer = 'random_normal')
        self.y = tf.keras.layers.Conv1D(filters = self.internal_units, kernel_size= 1, padding='same', kernel_initializer = 'random_normal')
        #self.g = tf.keras.layers.Conv1D(filters = self.internal_units, kernel_size= 1, padding='same')
        self.out_conv = tf.keras.layers.Conv1D(filters = self.units, kernel_size= 1, padding='same', kernel_initializer = 'random_normal')
        self.act = tf.keras.layers.Softmax(axis = 0)



    def call(self, inputs):
        #suppose to have as input (batch_size, length, channels)


        
        y_T = tf.transpose(self.y(inputs), perm = [0, 2, 1]) # (batch_size, channels, length)
        xy_T = self.act(tf.matmul(self.x(inputs), y_T)) # (batch_size, length, length) --> (32, 4500, 4500)


        #ext =  # (batch_size, length, channels)
        s_dot = tf.matmul(xy_T, self.out_conv(inputs)) # (batch_size, length, channels)

        return inputs + s_dot


class LearnablePositionalEncoding1D_ChannelsLast(tf.keras.layers.Layer):
    def __init__(self, length, embedding_dim, **kwargs):
        super(LearnablePositionalEncoding1D_ChannelsLast, self).__init__(**kwargs)
        self.length = length
        self.embedding_dim = embedding_dim
        self.pos_embedding = self.add_weight(
            shape=(1, length, embedding_dim),
            initializer='random_normal',
            trainable=True,
            name='pos_embedding'
        )

    def call(self, inputs):
        return inputs + self.pos_embedding  

    def get_config(self):
        config = super(LearnablePositionalEncoding1D_ChannelsLast, self).get_config()
        config.update({
            'length': self.length,
            'embedding_dim': self.embedding_dim
        })
        return config

 

class PeakPositionsv2(tf.keras.layers.Layer):
    def __init__(self, padding = 5, threshold = 20):
        super(PeakPositionsv2, self).__init__()
        self.padding = padding
        self.threshold = threshold
    """The output is made of spectra with only two values (0, 1000)."""
    
    def call(self, inputs):
        
        is_peak = tf.equal(inputs, inputs) # Init a full True tensor

        for i in range(1,self.padding):

            left_shift = tf.pad(inputs[:, :-i], [[0, 0], [i, 0], [0, 0]], mode='CONSTANT', constant_values=tf.reduce_min(inputs))
            right_shift = tf.pad(inputs[:, i:], [[0, 0], [0, i], [0, 0]], mode='CONSTANT', constant_values=tf.reduce_min(inputs))

            is_peak_var = tf.logical_and(tf.greater(inputs, left_shift), tf.greater(inputs, right_shift))
            
            is_peak = tf.logical_and(is_peak, is_peak_var)



        is_peak = tf.logical_and(is_peak, tf.greater(inputs, self.threshold))
        
        peak_mask = tf.cast(is_peak, dtype=tf.float16)
        peak_spectrum = inputs * (1 - peak_mask) + peak_mask * 1000
        
        batch_size = tf.shape(inputs)[0]
        spectra_len = tf.shape(inputs)[1]
        return tf.reshape(peak_spectrum, shape=(batch_size, spectra_len, 1))
