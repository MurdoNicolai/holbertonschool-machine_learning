#!/usr/bin/env python3
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

lib = np.load("pca.npz")
data = lib["data"]
labels = lib["labels"]

data_means = np.mean(data, axis=0)
norm_data = data - data_means
_, _, Vh = np.linalg.svd(norm_data)
pca_data = np.matmul(norm_data, Vh[:3].T)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
def randrange(n, vmin, vmax):
    '''
    Helper function to make an array of random numbers having shape (n, )
    with each number distributed Uniform(vmin, vmax).
    '''
    return (vmax - vmin)*np.random.rand(n) + vmin

n = 100

# For each set of style and range settings, plot n random points in the box
# defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].
xs = []
ys = []
zs = []
pos = 0
for dt in pca_data:
    xs.append(dt[0])
    ys.append(dt[1])
    zs.append(dt[2])
    pos += 1
ax.scatter(xs, ys, zs, c=labels, marker='o')
ax.set_title('PCA of Iris Dataset')
ax.set_xlabel('U1')
ax.set_ylabel('U2')
ax.set_zlabel('U3')

plt.show()

