#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4, 3))


people = ('Farrah', 'Fred', 'Felicia')
fruits = {'apples': np.array(fruit[0]),
          'bananas': np.array(fruit[1]),
          'oranges': np.array(fruit[2]),
          'peaches': np.array(fruit[3])}


fig, ax = plt.subplots()
colors = ('red', 'yellow', '#ff8000', '#ffe5b4')
colorkey = 0
bottom = np.zeros(3)
for fruit, value in fruits.items():
    p = ax.bar(people, value, label=fruit, width=0.5, bottom=bottom,
               color=colors[colorkey])
    bottom += value
    colorkey += 1
ax.set_title("Number of Fruit per Person")
ax.set_ylabel('Quantity of Fruit')
ax.legend(loc="upper right")
ax.set_ylim(0, 80)
plt.show()
