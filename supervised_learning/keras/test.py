#!/usr/bin/env python3

import tensorflow.keras as K
build_model = __import__('0-sequential').build_model

model = build_model(200, [100, 50, 10], ['tanh', 'sigmoid', 'softmax'], 0.01, 0.6)
for layer in model.layers:
    if type(layer) == K.layers.Dense:
        try:
            print(layer.kernel_regularizer.l1)
        except AttributeError:
            print(.0)
        print(layer.kernel_regularizer.l2)
