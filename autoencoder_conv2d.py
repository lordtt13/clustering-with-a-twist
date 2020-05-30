#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 31 05:03:52 2020

@author: tanmay
"""

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Input, Reshape, Flatten, Conv2DTranspose, Dense


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