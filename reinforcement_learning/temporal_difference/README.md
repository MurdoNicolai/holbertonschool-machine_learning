# Temporal Difference

## Master
By: Alexa Orrico, Software Engineer at Holberton School

## Resources

Read or watch:

- [RL Course by David Silver - Lecture 4: Model-Free Prediction](https://www.youtube.com/watch?v=PnHCvfgC_ZA)
- [RL Course by David Silver - Lecture 5: Model Free Control](https://www.youtube.com/watch?v=0g4j2k_Ggc4)
- [Simple Reinforcement Learning: Temporal Difference Learning](https://towardsdatascience.com/simple-reinforcement-learning-temporal-difference-learning-ec6781c5b376)
- [On-Policy TD Control](https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf)

Definitions to skim:

- [Monte Carlo method](https://en.wikipedia.org/wiki/Monte_Carlo_method)
- [Temporal difference learning](https://en.wikipedia.org/wiki/Temporal_difference_learning)
- [State–action–reward–state–action](https://en.wikipedia.org/wiki/State–action–reward–state–action)

## Learning Objectives

- What is Monte Carlo?
- What is Temporal Difference?
- What is bootstrapping?
- What is n-step temporal difference?
- What is TD(λ)?
- What is an eligibility trace?
- What is SARSA? SARSA(λ)? SARSAMAX?
- What is ‘on-policy’ vs ‘off-policy’?

## Requirements
### General

- Allowed editors: vi, vim, emacs
- All your files will be interpreted/compiled on Ubuntu 16.04 LTS using python3 (version 3.5)
- Your files will be executed with numpy (version 1.15), and gym (version 0.7)
- All your files should end with a new line
- The first line of all your files should be exactly `#!/usr/bin/env python3`
- A README.md file, at the root of the folder of the project, is mandatory
- Your code should use the pycodestyle style (version 2.4)
- All your modules should have documentation (`python3 -c 'print(__import__("my_module").__doc__)'`)
- All your classes should have documentation (`python3 -c 'print(__import__("my_module").MyClass.__doc__)'`)
- All your functions (inside and outside a class) should have documentation (`python3 -c 'print(__import__("my_module").my_function.__doc__)' and `python3 -c 'print(__import__("my_module").MyClass.my_function.__doc__)'`)
- All your files must be executable
- Your code should use the minimum number of operations

## Tasks

### 0. Monte Carlo

Write the function `def monte_carlo(env, V, policy, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99):` that performs the Monte Carlo algorithm:

- `env` is the openAI environment instance
- `V` is a numpy.ndarray of shape `(s,)` containing the value estimate
- `policy` is a function that takes in a state and returns the next action to take
- `episodes` is the total number of episodes to train over
- `max_steps` is the maximum number of steps per episode
- `alpha` is the learning rate
- `gamma` is the discount rate
- Returns: `V`, the updated value estimate

### 1. TD(λ)

Write the function `def td_lambtha(env, V, policy, lambtha, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99):` that performs the TD(λ) algorithm:

- `env` is the openAI environment instance
- `V` is a numpy.ndarray of shape `(s,)` containing the value estimate
- `policy` is a function that takes in a state and returns the next action to take
- `lambtha` is the eligibility trace factor
- `episodes` is the total number of episodes to train over
- `max_steps` is the maximum number of steps per episode
- `alpha` is the learning rate
- `gamma` is the discount rate
- Returns: `V`, the updated value estimate

### 2. SARSA(λ)

Write the function `def sarsa_lambtha(env, Q, lambtha, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99, epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):` that performs SARSA(λ):

- `env` is the openAI environment instance
- `Q` is a numpy.ndarray of shape `(s,a)` containing the Q table
- `lambtha` is the eligibility trace factor
- `episodes` is the total number of episodes to train over
- `max_steps` is the maximum number of steps per episode
- `alpha` is the learning rate
- `gamma` is the discount rate
- `epsilon` is the initial threshold for epsilon greedy
- `min_epsilon` is the minimum value that epsilon should decay to
- `epsilon_decay` is the decay rate for updating epsilon between episodes
- Returns: `Q`, the updated Q table
