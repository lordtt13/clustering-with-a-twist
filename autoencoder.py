#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 30 04:30:40 2020

@author: tanmay
"""

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense


def autoencoder(dims, act = 'relu', init = 'glorot_uniform'):
    """
    Fully connected auto-encoder model, symmetric.
    Arguments:
        dims: list of number of units in each layer of encoder. dims[0] is input dim, dims[-1] is units in hidden layer.
            The decoder is symmetric with encoder. So number of layers of the auto-encoder is 2*len(dims)-1
        act: activation, not applied to Input, Hidden and Output layers
    return:
        (ae_model, encoder_model), Model of autoencoder and model of encoder
    """
    n_stacks = len(dims) - 1
    # input
    input_img = Input(shape = (dims[0],), name = 'input')
    x = input_img
    # internal layers in encoder
    for i in range(n_stacks-1):
        x = Dense(dims[i + 1], activation = act, kernel_initializer = init, name = 'encoder_%d' % i)(x)

    # hidden layer
    encoded = Dense(dims[-1], kernel_initializer = init, name = 'encoder_%d' % (n_stacks - 1))(x)  # hidden layer, features are extracted from here

    x = encoded
    # internal layers in decoder
    for i in range(n_stacks-1, 0, -1):
        x = Dense(dims[i], activation = act, kernel_initializer = init, name = 'decoder_%d' % i)(x)

    # output
    x = Dense(dims[0], kernel_initializer = init, name = 'decoder_0')(x)
    decoded = x
    return Model(inputs = input_img, outputs = decoded, name = 'AE'), Model(inputs = input_img, outputs = encoded, name = 'encoder')