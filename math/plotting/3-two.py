#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0, 21000, 1000)
r = np.log(0.5)
t1 = 5730
t2 = 1600
y1 = np.exp((r / t1) * x)
y2 = np.exp((r / t2) * x)

plt.title("Exponential Decay of Radioactive Element")
plt.xlabel('Time (years)')
plt.ylabel('Fraction Remaining')
plt.plot(x, y1, 'r--')
plt.plot(x, y2, 'g')
plt.legend('upper right', labels=['C-14', 'Ra-226'])
plt.show()
