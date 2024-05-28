#!/usr/bin/env python3

def monte_carlo(env, V, policy, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99):
    """
    Perform the Monte Carlo algorithm for updating the value estimate.
    """
    for episode in range(episodes):
        # Generate an episode
        state = env.reset()
        episode_data = []

        for step in range(max_steps):
            action = policy(state)
            next_state, reward, done, _ = env.step(action)
            episode_data.append((state, reward))
            state = next_state
            if done:
                break

        # Calculate returns and update value function
        G = 0
        for state, reward in reversed(episode_data):
            G = reward + gamma * G
            # Update the value function using incremental mean
            V[state] += alpha * (G - V[state])

    return V
