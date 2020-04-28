import numpy as np
from scipy.special import expit
from scipy.sparse.linalg import svds

C = 1

class LogisticRegression:
    def __init__(self):
        self.w = None

    def train(self, x, y, epochs=5):
        n, d = x.shape
        if self.w is None:
            self.w = np.random.rand(d, 1)

        _, s, _ = svds(x, k=1)
        l_singular_val = s[0]
        # l_singular_val = np.sqrt(np.square(x).sum())

        for epoch in range(epochs):
            h = expit(x@self.w)
            acc = ((h > 0.5) == y).sum() / n
            cost_d = x.T @ (h - y) / n

            cost =  ((1-h[y==1]).sum() + h[y==0].sum())/n

            if epoch % 100 == 0:
                print(cost.sum(), acc)

            alpha = 4*n*C*np.linalg.norm(cost_d)/(l_singular_val**2)
            # print(alpha)
            # alpha = 0.1

            self.w -= alpha*cost_d

    def score(self, x, y):
        n, _ = x.shape
        if self.w is None:
            raise NotImplementedError
        h = expit(x @ self.w)
        acc = ((h > 0.5) == y).sum() / n
        return acc