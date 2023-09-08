import numpy as np

def dropout_forward_prop(X, weights, L, keep_prob):
    """
    Conduct forward propagation using Dropout.
    """

    cache = {}
    A = X

    for layer in range(1, L + 1):
        W = weights['W' + str(layer)]
        b = weights['b' + str(layer)]

        Z = np.dot(W, A) + b
        if layer < L:
            A = np.tanh(Z)
            D = np.random.rand(A.shape[0], A.shape[1]) < keep_prob
            A *= D
            A /= keep_prob
            cache['D' + str(layer)] = D
        else:
            A = np.exp(Z - np.max(Z)) / np.sum(np.exp(Z - np.max(Z)), axis=0, keepdims=True)

        cache['A' + str(layer)] = A

    return cache
