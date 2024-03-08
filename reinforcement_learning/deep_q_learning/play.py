https://cdn.discordapp.com/attachments/1014522471543742604/1215773582551158894/double_circular_linked_list.zip?ex=65fdf862&is=65eb8362&hm=54a6a322752c5df99a3c698d7c1f5336f5879793f2821c4d2c6a17ce40efd6db&#!/usr/bin/env python3
import gymnasium as gym
from keras.models import load_model
from gym.wrappers.atari_preprocessing import AtariPreprocessing


# Load the trained policy network
model = load_model("policy.h5")

# Environment setup
env = gym.make("BreakoutNoFrameskip-v4")
env = AtariPreprocessing(env).wrap()

# Play the game using the GreedyQPolicy
policy = GreedyQPolicy(model=model)
state = env.reset()
while True:
    action = policy.act(state)
    next_state, reward, done, _ = env.step(action)
    env.render()  # Render the game screen
    if done:
        break
    state = next_state

# Close the environment
env.close()
