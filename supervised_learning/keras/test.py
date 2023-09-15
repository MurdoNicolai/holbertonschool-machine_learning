#!/usr/bin/env python3


build_model = __import__('0-sequential').build_model

model = build_model(200, [100, 50, 10], ['tanh', 'sigmoid', 'softmax'], 0.01, 0.6)
model.summary()
