# Policy Gradients

## Master
By: Alexa Orrico, Software Engineer at Holberton School

## Resources

Read or watch:

- [How Policy Gradient Reinforcement Learning Works](https://medium.com/@leosimmons/how-policy-gradient-reinforcement-learning-works-8a1d1b13cfe4)
- [Policy Gradients in a Nutshell](https://towardsdatascience.com/policy-gradients-in-a-nutshell-77b79b7b9e05)
- [RL Course by David Silver - Lecture 7: Policy Gradient Methods](https://www.youtube.com/watch?v=KHZVXao4qXs)
- [Reinforcement Learning 6: Policy Gradients and Actor Critics](https://www.youtube.com/watch?v=tqrcjHuNdmQ)
- [Policy Gradient Algorithms](http://rail.eecs.berkeley.edu/deeprlcourse-fa17/f17docs/lecture_5_policy_gradient.pdf)

## Learning Objectives

- What is Policy?
- How to calculate a Policy Gradient?
- What and how to use a Monte-Carlo policy gradient?

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

### 0. Simple Policy function

Write a function that computes the policy with a weight of a matrix.

- Prototype: `def policy(matrix, weight):`
- `matrix` is a numpy.ndarray of shape `(1, n)` containing the state
- `weight` is a numpy.ndarray of shape `(n, m)` containing the weights
- Returns: a numpy.ndarray of shape `(1, m)` containing the policy

### 1. Compute the Monte-Carlo policy gradient

By using the previous function created `policy`, write a function that computes the Monte-Carlo policy gradient based on a state and a weight matrix.

- Prototype: `def policy_gradient(state, weight):`
  - `state`: matrix representing the current observation of the environment
  - `weight`: matrix of random weights
- Returns: the action and the gradient (in this order)

### 2. Implement the training

By using the previous function created `policy_gradient`, write a function that implements a full training.

- Prototype: `def train(env, nb_episodes, alpha=0.000045, gamma=0.98):`
  - `env`: initial environment
  - `nb_episodes`: number of episodes used for training
  - `alpha`: the learning rate
  - `gamma`: the discount factor
- Returns: all values of the score (sum of all rewards during one episode loop)

Since the training is quite long, please print the current episode number and the score after each loop. To display this information on the same line, you can use `end="\r", flush=True` of the print function.

### 3. Animate iteration

Update the prototype of the `train` function by adding a last optional parameter `show_result` (default: `False`).

When this parameter is `True`, render the environment every 1000 episodes computed.
