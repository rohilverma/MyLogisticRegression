import numpy as np
from scipy.special import expit
from collections.abc import Iterable
from matplotlib import pyplot as plt


# Create problem / dataset
d = 10000
w = np.random.rand(d, 1)*2-1   # transform to -1-1

n = 100000
x = np.random.rand(n, d)*2-1
xw = expit(x@w)
y = (xw > 0.5).astype(int)

plt.scatter(range(n), y, c=y)
plt.show()
plt.clf()

# Import classifier + initialize parameters, optimizer
from LogisticRegression import LogisticRegression
lr = LogisticRegression()
lr.train(x, y, 201)

h = expit(x@lr.w)
# print(w, lr.w)
preds = h > 0.5

plt.scatter(range(n), preds, c=y)
plt.show()

# Train model

# Evaluate model