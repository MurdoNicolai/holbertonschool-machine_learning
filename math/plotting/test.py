#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

np.random.seed(5)
nb_throws = 10000
t1=[]
t2=[]
t3=[]
t4=[]
t5=[]
t6=[]
t7=[]
t8=[]
t9=[]
t10=[]
for throw in range(nb_throws):
    dice = np.random.randint(100, size=10)
    dice = sorted(dice, reverse=False)
    if throw == 10:
        print(dice)
    t1.append(dice[0])
    t2.append(dice[1])
    t3.append(dice[2])
    t4.append(dice[3])
    t5.append(dice[4])
    t6.append(dice[5])
    t7.append(dice[6])
    t8.append(dice[7])
    t9.append(dice[8])
    t10.append(dice[9])

x = range(100)

density = stats.gaussian_kde(t1)
plt.plot(x, density(x), 'b')
density = stats.gaussian_kde(t2)
plt.plot(x, density(x), 'r')
density = stats.gaussian_kde(t3)
plt.plot(x, density(x), 'g')
density = stats.gaussian_kde(t4)
plt.plot(x, density(x), 'b')
density = stats.gaussian_kde(t5)
plt.plot(x, density(x), 'r')
density = stats.gaussian_kde(t6)
plt.plot(x, density(x), 'g')
density = stats.gaussian_kde(t7)
plt.plot(x, density(x), 'b')
density = stats.gaussian_kde(t8)
plt.plot(x, density(x), 'r')
density = stats.gaussian_kde(t9)
plt.plot(x, density(x), 'g')
density = stats.gaussian_kde(t10)
plt.plot(x, density(x), 'b')

plt.show()
