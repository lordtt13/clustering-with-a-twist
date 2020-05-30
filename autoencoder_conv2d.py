#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 31 05:03:52 2020

@author: tanmay
"""

import numpy as np

from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Conv2D, Input, Reshape, Flatten, Conv2DTranspose, Dense, UpSampling2D


def autoencoderConv2D_1(input_shape = (28, 28, 1), filters = [32, 64, 128, 10]):
    input_img = Input(shape = input_shape)
    if input_shape[0] % 8 == 0:
        pad3 = 'same'
    else:
        pad3 = 'valid'
    x = Conv2D(filters[0], 5, strides = 2, padding = 'same', activation = 'relu', name = 'conv1', input_shape = input_shape)(input_img)

    x = Conv2D(filters[1], 5, strides = 2, padding = 'same', activation = 'relu', name = 'conv2')(x)

    x = Conv2D(filters[2], 3, strides = 2, padding = pad3, activation = 'relu', name = 'conv3')(x)

    x = Flatten()(x)
    encoded = Dense(units = filters[3], name = 'embedding')(x)
    x = Dense(units = filters[2]*int(input_shape[0]/8)*int(input_shape[0]/8), activation = 'relu')(encoded)

    x = Reshape((int(input_shape[0]/8), int(input_shape[0]/8), filters[2]))(x)
    x = Conv2DTranspose(filters[1], 3, strides = 2, padding = pad3, activation = 'relu', name = 'deconv3')(x)

    x = Conv2DTranspose(filters[0], 5, strides = 2, padding = 'same', activation = 'relu', name = 'deconv2')(x)

    decoded = Conv2DTranspose(input_shape[2], 5, strides = 2, padding = 'same', name = 'deconv1')(x)
    return Model(inputs = input_img, outputs = decoded, name = 'AE'), Model(inputs = input_img, outputs = encoded, name = 'encoder')


def autoencoderConv2D_2(img_shape = (28, 28, 1)):
    """
    Conv2D auto-encoder model.
    Arguments:
        img_shape: e.g. (28, 28, 1) for MNIST
    return:
        (autoencoder, encoder), Model of autoencoder and model of encoder
    """
    input_img = Input(shape = img_shape)
    # Encoder
    x = Conv2D(16, (3, 3), activation = 'relu', padding = 'same', strides = (2, 2))(input_img)
    x = Conv2D(8, (3, 3), activation = 'relu', padding = 'same', strides = (2, 2))(x)
    x = Conv2D(8, (3, 3), activation = 'relu', padding = 'same', strides = (2, 2))(x)
    shape_before_flattening = K.int_shape(x)
    # at this point the representation is (4, 4, 8) i.e. 128-dimensional
    x = Flatten()(x)
    encoded = Dense(10, activation = 'relu', name = 'encoded')(x)

    # Decoder
    x = Dense(np.prod(shape_before_flattening[1:]),
                activation = 'relu')(encoded)
    # Reshape into an image of the same shape as before our last `Flatten` layer
    x = Reshape(shape_before_flattening[1:])(x)

    x = Conv2D(8, (3, 3), activation = 'relu', padding = 'same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(8, (3, 3), activation = 'relu', padding = 'same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), activation = 'relu')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(1, (3, 3), activation = 'sigmoid', padding = 'same')(x)

    return Model(inputs = input_img, outputs = decoded, name = 'AE'), Model(inputs = input_img, outputs = encoded, name = 'encoder')