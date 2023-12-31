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


def stddeviation(list):
    """returns the stddev of list"""
    mean = average(list)
    sum = 0
    for num in list:
        sum += (num - mean)*(num - mean)
    return ((sum/len(list))**(1/2))


def factorial(num):
    """returns the factorial ofnum"""
    result = 1
    for i in range(num):
        result *= i+1
    return result


class Normal():
    """class representing the Normal distribution"""
    def __init__(self, data=None, mean=0., stddev=1.):
        """data is list of data to estimate distribution
        stddev is expect number of occurences:
        sets stddev"""
        if data is None:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
            else:
                self.stddev = float(stddev)
                self.mean = float(mean)
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            elif len(data) <= 2:
                raise ValueError("data must contain multiple values")
            else:
                self.mean = average(data)
                self.stddev = stddeviation(data)

    def pdf(self, k):
        """calc the pmf given the amount of successes"""
        p1 = 1/(self.stddev*(2*π())**(1/2))
        p2 = ((k-self.mean)/self.stddev)**2
        p3 = pow(e(), p2/2)
        return p1/p3

    def cdf(self, k):
        """calc the cdf given the amount of successes"""
        return ((1 + erf((k-self.mean)/(self.stddev*(2**(1/2)))))/2)

    def z_score(self, x):
        """calc the z-scores"""
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """calc the x_value"""
        return self.mean + z * self.stddev
