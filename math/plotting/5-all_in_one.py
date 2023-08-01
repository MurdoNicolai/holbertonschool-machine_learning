#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

y0 = np.arange(0, 11) ** 3

mean = [69, 0]
cov = [[15, 8], [8, 15]]
np.random.seed(5)
x1, y1 = np.random.multivariate_normal(mean, cov, 2000).T
y1 += 180

x2 = np.arange(0, 28651, 5730)
r2 = np.log(0.5)
t2 = 5730
y2 = np.exp((r2 / t2) * x2)

x3 = np.arange(0, 21000, 1000)
r3 = np.log(0.5)
t31 = 5730
t32 = 1600
y31 = np.exp((r3 / t31) * x3)
y32 = np.exp((r3 / t32) * x3)

np.random.seed(5)
student_grades = np.random.normal(68, 15, 50)

plt.subplot(321)
plt.plot(np.arange(0, 11), y0, 'r')


plt.subplot(322)
plt.title("Men's Height vs Weight", fontsize='x-small')
plt.xlabel('Height (in)', fontsize='x-small')
plt.ylabel('Width (lbs)')
plt.plot(x1, y1, '.m')


plt.subplot(323)
plt.title("Exponential Decay of C-14", fontsize='x-small')
plt.xlabel('Time (years)', fontsize='x-small')
plt.ylabel('Fraction Remaining', fontsize='x-small')
plt.plot(x2, y2)
plt.semilogy()

plt.subplot(324)
plt.xlabel('Time (years)', fontsize='x-small')
plt.ylabel('Fraction Remaining', fontsize='x-small')
plt.plot(x3, y31, 'r--')
plt.plot(x3, y32, 'g')
plt.legend('upper right', labels=['C-14', 'Ra-226'])

plt.subplot(3, 2, (5, 6))
plt.title("Project A", fontsize='x-small')
plt.xlabel('Grades', fontsize='x-small')
plt.ylabel('Number of Students', fontsize='x-small')
plt.hist(x=student_grades, bins=range(10, 110, 10), edgecolor='black')

plt.subplots_adjust(wspace=(0.5), hspace=(0.5))
plt.suptitle('All in One')
plt.show()
