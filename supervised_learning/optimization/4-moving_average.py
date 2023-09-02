#!/usr/bin/env python3
""" contains normalization"""
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()


def moving_average(data, beta):
    """ calculates moving average"""
    Average = 0
    total_numbers = 0
    moving_average = []
    for number in data:
        total_numbers += 1
        print((beta * Average), ((1 - beta) * number),
              (1 - beta**total_numbers))
        Average = ((beta * Average) + ((1 - beta) * number))

        moving_average.append(Average/(1 - beta ** total_numbers))
    return moving_average
