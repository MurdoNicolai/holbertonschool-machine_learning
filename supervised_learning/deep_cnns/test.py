#!/usr/bin/env python3

import tensorflow.keras as K
identity_block = __import__('2-identity_block').identity_block

X1 = K.Input(shape=(56, 56, 256))
Y1 = identity_block(X1, [64, 64, 256])
model1 = K.models.Model(inputs=X1, outputs=Y1)
for layer in model1.layers:
    if type(layer) is K.layers.Conv2D:
        for k, v in sorted(layer.kernel_initializer.__dict__.items()):
            if k == "_random_generator":
                print(k, type(v))
            else:
                print(k, v)
X2 = K.Input(shape=(28, 28, 512))
Y2 = identity_block(X2, [128, 128, 512])
model2 = K.models.Model(inputs=X2, outputs=Y2)
for layer in model2.layers:
    if type(layer) is K.layers.Conv2D:
        for k, v in sorted(layer.kernel_initializer.__dict__.items()):
            if k == "_random_generator":
                print(k, type(v))
            else:
                print(k, v)
