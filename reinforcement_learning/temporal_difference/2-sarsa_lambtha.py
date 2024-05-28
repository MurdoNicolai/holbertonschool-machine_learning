#!/usr/bin/env python3
import numpy as np


def epsilon_greedy_policy(Q, state, epsilon):
    """
    Epsilon-greedy policy for action selection.
    """
    if np.random.rand() < epsilon:
        return np.random.choice(len(Q[state]))
    else:
        return np.argmax(Q[state])

def sarsa_lambtha(env, Q, lambtha, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99, epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """
    Perform the SARSA(λ) algorithm for updating the Q table.
    """
    for episode in range(episodes):
        state = env.reset()
        action = epsilon_greedy_policy(Q, state, epsilon)
        E = np.zeros_like(Q)  # Initialize eligibility traces

        for step in range(max_steps):
            next_state, reward, done, _ = env.step(action)
            next_action = epsilon_greedy_policy(Q, next_state, epsilon)
            # Compute the TD error
            td_error = reward + gamma * Q[next_state, next_action] * (1 - done) - Q[state, action]
            # Update eligibility trace for the current state-action pair
            E[state, action] += 1

            # Update the Q table and eligibility traces
            Q += alpha * td_error * E
            E *= gamma * lambtha

            if done:
                break

            state = next_state
            action = next_action

        # Decay epsilon
        epsilon = max(min_epsilon, epsilon * (1 - epsilon_decay))

    return Q
