import tensorflow as tf
import numpy as np


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



 

def conv_1D(input_shape, output_shape, seed = seed, alpha = 0.01, learning_rate = 0.001):


    input_layer = tf.keras.layers.Input(shape = input_shape, name ="Input")
    pool1 = tf.keras.layers.MaxPooling1D(pool_size = 3)(input_layer)

    conv1 = tf.keras.layers.Conv1D(kernel_size= 3, filters = 5, activation= 'relu', padding= 'same')(pool1)
    conv2 = tf.keras.layers.Conv1D(kernel_size= 3, filters = 5, activation= 'relu', padding= 'same')(conv1)
    #pool2 = tfkl.MaxPooling1D(pool_size = 2, name = "Pool_2")(conv2)

    conv3 = tf.keras.layers.Conv1D(kernel_size= 3, filters = 10, activation= 'relu', padding= 'same')(conv2)
    conv4 = tf.keras.layers.Conv1D(kernel_size= 3, filters = 10,activation= 'relu', padding= 'same')(conv3)

    pool3 = tf.keras.layers.MaxPooling1D(pool_size = 2, name = "Pool3")(conv4)
    x = NonLocalBlock(units =10)(pool3)

    conv5 = tf.keras.layers.Conv1D(kernel_size= 5, filters = 15, activation= 'relu', padding= 'same')(x)
    conv6 = tf.keras.layers.Conv1D(kernel_size= 5, filters = 15, activation= 'relu', padding= 'same')(conv5)
    pool4 = tf.keras.layers.MaxPooling1D(pool_size = 3, name = "Pool4")(conv6)

    conv7 = tf.keras.layers.Conv1D(kernel_size= 5, filters = 20, activation= 'relu', padding= 'same')(pool4)
    conv8 = tf.keras.layers.Conv1D(kernel_size= 5, filters = 20, activation= 'relu', padding= 'same')(conv7)
    pool5 = tf.keras.layers.MaxPooling1D(pool_size = 2, name = "Pool5")(conv8)

    conv9 = tf.keras.layers.Conv1D(kernel_size= 5, filters = 30, activation= 'relu', padding= 'same')(pool5)
    conv10 = tf.keras.layers.Conv1D(kernel_size= 5, filters = 30, activation= 'relu', padding= 'same')(conv9)
    pool6 = tf.keras.layers.MaxPooling1D(pool_size = 5, name = "Pool6")(conv10)

    flattening_layer = tf.keras.layers.Flatten()(pool6)

    dense_1 = tf.keras.layers.Dense(units = 256, activation= 'relu', kernel_regularizer= tf.keras.regularizers.L2(alpha))(flattening_layer)
    dropout1 = tf.keras.layers.Dropout(0.1)(dense_1)

    dense_2 = tf.keras.layers.Dense(units = 80, activation= 'relu', kernel_regularizer= tf.keras.regularizers.L2(alpha))(dropout1)
    dropout2 = tf.keras.layers.Dropout(0.1)(dense_2)

    dense_3 = tf.keras.layers.Dense(units = 128, activation= 'relu', kernel_regularizer= tf.keras.regularizers.L2(alpha))(dropout2)
    output_layer = tf.keras.layers.Dense(units = output_shape)(dense_3)


    model = tf.keras.Model(inputs = input_layer, outputs = output_layer, name = "conv_1D_10NLB")

    model.compile(loss = tf.keras.losses.MeanAbsolutePercentageError(), optimizer = tf.keras.optimizers.legacy.Adam(learning_rate = learning_rate), metrics = ['mean_absolute_percentage_error'])

    return model



def conv_1DnoNLB(input_shape, output_shape, seed = seed, alpha = 0.01, learning_rate = 0.001):


    input_layer = tf.keras.layers.Input(shape = input_shape, name ="Input")
    pool1 = tf.keras.layers.MaxPooling1D(pool_size = 3)(input_layer)

    conv1 = tf.keras.layers.Conv1D(kernel_size= 3, filters = 5, activation= 'relu', padding= 'same')(pool1)
    conv2 = tf.keras.layers.Conv1D(kernel_size= 3, filters = 5, activation= 'relu', padding= 'same')(conv1)
    #pool2 = tfkl.MaxPooling1D(pool_size = 2, name = "Pool_2")(conv2)

    conv3 = tf.keras.layers.Conv1D(kernel_size= 3, filters = 10, activation= 'relu', padding= 'same')(conv2)
    conv4 = tf.keras.layers.Conv1D(kernel_size= 3, filters = 10,activation= 'relu', padding= 'same')(conv3)

    pool3 = tf.keras.layers.MaxPooling1D(pool_size = 2, name = "Pool3")(conv4)
    #x = NonLocalBlock(units =10)(pool3)

    conv5 = tf.keras.layers.Conv1D(kernel_size= 5, filters = 15, activation= 'relu', padding= 'same')(pool3)
    conv6 = tf.keras.layers.Conv1D(kernel_size= 5, filters = 15, activation= 'relu', padding= 'same')(conv5)
    pool4 = tf.keras.layers.MaxPooling1D(pool_size = 3, name = "Pool4")(conv6)

    conv7 = tf.keras.layers.Conv1D(kernel_size= 5, filters = 20, activation= 'relu', padding= 'same')(pool4)
    conv8 = tf.keras.layers.Conv1D(kernel_size= 5, filters = 20, activation= 'relu', padding= 'same')(conv7)
    pool5 = tf.keras.layers.MaxPooling1D(pool_size = 2, name = "Pool5")(conv8)

    conv9 = tf.keras.layers.Conv1D(kernel_size= 5, filters = 30, activation= 'relu', padding= 'same')(pool5)
    conv10 = tf.keras.layers.Conv1D(kernel_size= 5, filters = 30, activation= 'relu', padding= 'same')(conv9)
    pool6 = tf.keras.layers.MaxPooling1D(pool_size = 5, name = "Pool6")(conv10)

    flattening_layer = tf.keras.layers.Flatten()(pool6)

    dense_1 = tf.keras.layers.Dense(units = 256, activation= 'relu', kernel_regularizer= tf.keras.regularizers.L2(alpha))(flattening_layer)
    dropout1 = tf.keras.layers.Dropout(0.1)(dense_1)

    dense_2 = tf.keras.layers.Dense(units = 80, activation= 'relu', kernel_regularizer= tf.keras.regularizers.L2(alpha))(dropout1)
    dropout2 = tf.keras.layers.Dropout(0.1)(dense_2)

    dense_3 = tf.keras.layers.Dense(units = 128, activation= 'relu', kernel_regularizer= tf.keras.regularizers.L2(alpha))(dropout2)
    output_layer = tf.keras.layers.Dense(units = output_shape)(dense_3)


    model = tf.keras.Model(inputs = input_layer, outputs = output_layer, name = "conv_1D_10NLB")

    model.compile(loss = tf.keras.losses.MeanAbsolutePercentageError(), optimizer = tf.keras.optimizers.legacy.Adam(learning_rate = learning_rate), metrics = ['mean_absolute_percentage_error'])

    return model




