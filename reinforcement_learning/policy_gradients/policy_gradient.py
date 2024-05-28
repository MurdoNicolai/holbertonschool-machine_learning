#!/usr/bin/env python3
import numpy as np

def policy(matrix, weight):
    """
    Compute the policy using a weight matrix.
    """
    z = np.dot(matrix, weight)
    exp = np.exp(z - np.max(z))  # for numerical stability
    return exp / np.sum(exp)

def policy_gradient(state, weight):
    """
    Compute the policy gradient based on a state and weight matrix.
    grad: the gradient of the log-policy with respect to the weights.
    """
    # Get the action probabilities
    action_probs = policy(state, weight)

    # Sample an action from the action probability distribution
    action = np.random.choice(len(action_probs), p=action_probs)

    # Compute the gradient of the log-policy
    grad = np.zeros_like(weight)
    for i in range(len(action_probs)):
        if i == action:
            grad[:, i] = state * (1 - action_probs[i])
        else:
            grad[:, i] = -state * action_probs[i]

    return action, grad
