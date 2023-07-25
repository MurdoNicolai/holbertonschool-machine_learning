#!/usr/bin/env python3

"""contains add function"""


def add_arrays(arr1, arr2):
    """add arrays elem by elem"""
    if len(arr1) != len(arr2):
        return None
    newarr = []
    for position in range(len(arr1)):
        newarr.append(arr1[position]+arr2[position])
    return newarr
