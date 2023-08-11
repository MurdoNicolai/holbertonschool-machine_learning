#!/usr/bin/env python3
"""contains Normal class"""


def π():
    """return aprox pi"""
    return (3.1415926536)


def e():
    """return aprox e"""
    return(2.7182818285)


def erf(x):
    """return aprox erf(x)"""
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


def factorial(num):
    """returns the factorial ofnum"""
    result = 1
    for i in range(num):
        result *= i+1
    return result


class Binomial():
    """class representing the Binomial distribution"""
    def __init__(self, data=None, n=1, p=0.5):
        """data is list of data to estimate distribution
        stddev is expect number of occurences:
        sets stddev"""
        if data is None:
            if n <= 0:
                raise ValueError("n must be a positive value")
            if p <= 0 or p >= 1:
                raise ValueError("p must be greater than 0 and less than 1")
            else:
                self.p = float(p)
                self.n = n
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            elif len(data) <= 2:
                raise ValueError("data must contain multiple values")
            else:
                self.p = 1-(varience(data)/average(data))
                self.n = round(average(data)/self.p)
                self.p = average(data)/self.n

    def pmf(self, k):
        """calc the pmf given the amount of successes"""
        if type(k) is not int:
            k = int(k)
        if k < 0:
            return 0
        p1 = factorial(self.n)/(factorial(k)*factorial(self.n-k))
        p2 = pow(self.p, k)*pow(1-self.p, self.n-k)
        return p1*p2

    def cdf(self, k):
        """calc the cdf given the amount of successes"""
        if type(k) is not int:
            k = int(k)
        if k < 0:
            return 0
        result = 0
        for n in range(k + 1):
            result += self.pmf(n)
        return result
