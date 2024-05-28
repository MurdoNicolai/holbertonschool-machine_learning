#!/usr/bin/env python3
import numpy as np
from policy_gradient import policy_gradient

def train(env, nb_episodes, alpha=0.000045, gamma=0.98, show_result=False):
    """
    Train the policy using policy gradient.
    """
    # Initialize the weight matrix with small random values
    weight = np.random.rand(env.observation_space.shape[0], env.action_space.n) * 0.01

    scores = []

    for episode in range(nb_episodes):
        state = env.reset()
        done = False
        episode_rewards = []

        # Run an episode
        while not done:
            if show_result and episode % 1000 == 0:
                env.render()

            action, grad = policy_gradient(state, weight)
            next_state, reward, done, _ = env.step(action)

            episode_rewards.append(reward)

            # Compute the discounted reward
            for t in range(len(episode_rewards)):
                Gt = sum([gamma ** i * r for i, r in enumerate(episode_rewards[t:])])
                weight += alpha * Gt * grad

            state = next_state

        score = sum(episode_rewards)
        scores.append(score)

        print(f"Episode {episode + 1}/{nb_episodes}, Score: {score}", end="\r", flush=True)

    return scores
