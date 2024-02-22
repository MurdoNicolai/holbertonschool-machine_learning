#!/usr/bin/env python3
"""function for basic q learning"""
import gymnasium as gym

def load_frozen_lake(desc=None, map_name=None, is_slippery=False):
    """load environment: FrozenLakeEnv"""
    env = gym.make('FrozenLake-v1',
                   desc=desc,
                   map_name=map_name,
                   is_slippery=is_slippery,
                   render_mode="rgb_array")
    return env
