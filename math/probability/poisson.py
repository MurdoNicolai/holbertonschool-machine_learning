#!/usr/bin/env python3
"""contains poisson class"""


def π():
    """pi"""
    return (3.1415926536)


def e():
    """e"""
    return(2.7182818285)


def erf(x):
    """erf(x)"""
    return((2/1.77245385091) *
           (x-x*x*x/3+x*x*x*x*x/10-x*x*x*x*x*x*x/42+x*x*x*x*x*x*x*x*x/216))


def average(list):
    """returns the average of list"""
    sum = 0
    for num in list:
        sum += num
    return sum/len(list)


def varience(list):
    """returns the average of list"""
    mean = average(list)
    sum = 0
    for num in list:
        sum += (num - mean)*(num - mean)
    return sum/len(list)


class Poisson():
    """class representing the Poisson distribution"""
    def __init__(self, data=None, lambtha=1.):
        """data is list of data to estimate distribution
        lambtha is expect number of occurences:
        sets lambtha"""
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            else:
                self.lambtha = lambtha
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            elif len(data) <= 2:
                raise ValueError("data must contain multiple values")
            else:
                self.lambtha = average(data)
