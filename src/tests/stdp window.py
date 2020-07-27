from copy import deepcopy

import numpy as np
from matplotlib import pyplot as plt


def stdp_window(t, w, A, B, t1, t2):
    tau = 1 / (1 / t1 + 1 / t2)
    if t > 0:
        return (1/ 2) * B**2 * t1 * np.exp(-2 * np.abs(t) / t1) * w - tau * A * B * np.exp(-np.abs(t) / t1)
    if t <= 0:
        return (1 / 2) * B**2 * t1 * w - tau * A * B * np.exp(-np.abs(t) / t2)

    # if t > 0:
    #     return B**2 * np.exp(-2 * np.abs(t) / t1) * w - A * B * np.exp(- np.abs(t) / t1)
    # if t <= 0:
    #     return B**2 * w - A * B * np.exp(- np.abs(t) / t2)

for i in range(50):
    A = 100 * np.random.rand()
    B = 10 * np.random.rand()
    t1 = 10 * np.random.rand()
    t2 = 10 * np.random.rand()
    w = np.random.rand()

    t = np.linspace(-1, 1, 200)
    res = []
    for j in range(len(t)):
        res.append(deepcopy(stdp_window(t[j], w, A, B, t1, t2)))
    plt.plot(t, res)
    plt.show(block = True)
    plt.close()