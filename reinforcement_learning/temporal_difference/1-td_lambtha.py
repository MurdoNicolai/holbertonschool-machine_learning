#!/usr/bin/env python3
import numpy as np

def td_lambtha(env, V, policy, lambtha, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99):
    """
    Perform the TD(λ) algorithm for updating the value estimate.

    Returns:
    V: The updated value estimate.
    """
    for episode in range(episodes):
        state = env.reset()
        E = np.zeros_like(V)  # Initialize eligibility traces

        for step in range(max_steps):
            action = policy(state)
            next_state, reward, done, _ = env.step(action)
            # Compute the TD error
            td_error = reward + gamma * V[next_state] * (1 - done) - V[state]
            # Update eligibility trace for the current state
            E[state] += 1

            # Update the value function and eligibility traces
            V += alpha * td_error * E
            E *= gamma * lambtha

            if done:
                break

            state = next_state

    return V
