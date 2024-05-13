#!/usr/bin/env python3
''' contains blue determining functions'''
"""function for basic q learning"""
import gymnasium as gym
import numpy as np

def load_frozen_lake(desc=None, map_name=None, is_slippery=False):
    """load environment: FrozenLakeEnv"""
    env = gym.make('FrozenLake-v1',
                   desc=desc,
                   map_name=map_name,
                   is_slippery=is_slippery)
    return env


def q_init(env):
    """ initializes the Q-table"""
    observations = env.observation_space.n
    actions = env.action_space.n
    q = np.zeros((observations, actions))

    return q

def epsilon_greedy(Q, state, epsilon):
    """defines the next action using the epsilon greedy algorithm"""
    p = np.random.random()
    if p < epsilon:
      action = np.random.choice(Q.shape[1])
    else:
      action = np.argmax(Q[state])
    return action
