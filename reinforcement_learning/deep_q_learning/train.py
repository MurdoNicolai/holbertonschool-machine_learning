#!/usr/bin/env python3

import gymnasium as gym
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from keras_rl.agents.dqn import DQNAgent
from keras_rl.agents.policy import EpsGreedyQPolicy
from keras_rl.memory import SequentialMemory
from gym.wrappers.atari_preprocessing import AtariPreprocessing


# Environment setup
env = gym.make("BreakoutNoFrameskip-v4")
env = AtariPreprocessing(env).wrap()

# Define neural network architecture
model = Sequential()
model.add(Conv2D(32, (8, 8), strides=4, activation='relu', input_shape=(env.observation_space.shape)))
model.add(Conv2D(64, (4, 4), strides=2, activation='relu'))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(env.action_space.n, activation='linear'))

# Define the agent
policy = EpsGreedyQPolicy(model=model, eps=0.1, decay=0.99)
memory = SequentialMemory(limit=50000)
dqn = DQNAgent(model=model, policy=policy, memory=memory, nb_actions=env.action_space.n, nb_steps_warmup=50000)
dqn.compile(loss='mse', optimizer='adam')

# Training loop
total_episodes = 100
for episode in range(total_episodes):
    state = env.reset()
    action = dqn.act(state)
    next_state, reward, done, _ = env.step(action)
    dqn.remember(state, action, reward, next_state, done)
    while not done:
        action = dqn.act(next_state)
        next_state, reward, done, _ = env.step(action)
        dqn.remember(next_state, action, reward, next_state, done)
    dqn.fit(num_steps=32, visualize=False)
    if episode % 10 == 0:
        print(f"Episode {episode}/{total_episodes}: {dqn.test()}")

# Save the trained policy network
dqn.save_weights("policy.h5")

# Close the environment
env.close()
