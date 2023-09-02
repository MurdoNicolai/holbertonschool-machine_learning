#!/usr/bin/env python3
""" moving average"""
import numpy as np



def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """
    Update the learning rate using inverse time decay.
    """
    updated_alpha = alpha / (1 + decay_rate * (global_step // decay_step))
    return updated_alpha
