#!/usr/bin/env python3
"""early stopping"""


def early_stopping(cost, opt_cost, threshold, patience, count):
    """
    determines if you should stop gradient descent early
    """
    if abs(cost - opt_cost) < threshold:
        count += 1
    else:
        count = 0
    return (count >= patience, count)
