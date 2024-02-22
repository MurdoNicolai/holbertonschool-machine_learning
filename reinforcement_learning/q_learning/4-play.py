#!/usr/bin/env python3
"""function for basic q learning"""
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """defines the next action using the epsilon greedy algorithm"""
    p = np.random.random()
    if p < epsilon:
      action = np.random.choice(Q.shape[1])
    else:
      action = np.argmax(Q[state[0]])
    return action

def train(env, Q, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99,
          epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """performs Q-learning"""
    total_rewards = []
    for episode in range(episodes):
        if epsilon > min_epsilon:
            epsilon -= epsilon_decay
        state = env.reset()
        nb_action_in_Q = np.zeros(Q.shape)
        new_Q = np.zeros(Q.shape)
        reward = 0
        for nb_step in range(max_steps):
            action = epsilon_greedy(Q, state, epsilon)
            new_Q[state[0]][action] += 1
            nb_action_in_Q[state[0]][action] += 1
            state = env.step(action)

            if state[1] or state[2]:
                nb_action_in_Q[nb_action_in_Q == 0] = 1
                if state[1]: # arrrived at goal
                    reward = 1
                elif state[3]: # did max steps
                    break
                else: #fell in a hole
                    Q[state[0]][action] = -1
                    reward = -1
                    break

                new_q = ((new_Q/nb_action_in_Q) * reward)
                for i in range(Q.shape[0]):
                    for j in range(Q.shape[1]):
                        if new_q[i][j] == 0.0:
                            pass
                        else:
                            Q[i][j] = (Q[i][j] *(1-alpha)) + (new_q[i][j] * alpha)
                break
            new_Q = new_Q * gamma
        total_rewards.append(reward)
    return(Q, total_rewards)

def play(env, Q, max_steps=100):
    state = env.reset()
    for step in range(max_steps):
        action = np.argmax(Q[state[0]])
        state = env.step(action)
        plt.imshow(env.render())
        plt.show()
        if state[2]:
            break
    return (state[1])
