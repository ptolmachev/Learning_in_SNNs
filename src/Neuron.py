import numpy as np
from copy import deepcopy
from matplotlib import pyplot as plt

class Neuron:
    def __init__(self, type, dt = 0.1):
        self.type = type
        self.dt = dt
        self.v = 0
        self.spike_occurred = False # a variable which is only True when the spike (reset) has happened

    def rhs(self):
        return None

    def update(self):
        return None


class IzhikevichNeuron(Neuron):
    def __init__(self, a=0.02, b=0.2, d = 8, I = 4.0, thresh_v = 30, reset_v=-65, dt=0.1):
        super().__init__(type="IzhikevichNeuron", dt = dt)
        self.v = -55 + 20 * np.random.randn()
        self.I = 2.5 + 1 * np.random.randn()
        self.u = 1 + np.random.randn()
        self.a = a
        self.b = b
        self.d = d
        self.thresh_v = thresh_v
        self.reset_v = reset_v
        self.I_syn = 0


    def rhs(self):
        rhs_v = 0.04 * self.v**2 + 5 * self.v + 140 - self.u + self.I
        rhs_u = self.a * (self.b * self.v - self.u)
        return np.array([rhs_v, rhs_u])

    def step(self):
        self.spike_occurred = False
        RHS = self.rhs()
        self.v = self.v + self.dt * RHS[0]
        self.u = self.u + self.dt * RHS[1]
        if self.v > self.thresh_v:
            self.v = self.reset_v
            self.u = self.u + self.d
            self.spike_occurred = True
        return None


class LIFNeuron(Neuron):
    def __init__(self, a = 0.01, thresh_v = -20, reset_v=-65, dt=0.1):
        super().__init__(type="LifNeuron", dt = dt)
        self.v = -55 + 20 * np.random.randn()
        self.a = a
        self.I = 0.1 + 0.1 * np.random.randn()
        self.thresh_v = thresh_v
        self.reset_v = reset_v


    def rhs(self):
        rhs_v = -self.a * self.v
        return np.array([rhs_v])

    def step(self):
        self.spike_occurred = False
        RHS = self.rhs()
        self.v = self.v + self.dt * RHS[0]
        if self.v > self.thresh_v:
            self.v = self.reset_v
            self.spike_occurred = True
        return None


if __name__ == '__main__':
    dt = 0.1
    IN = IzhikevichNeuron(I=6, dt = dt)
    v = []
    for i in range(int(100 / dt)):
        IN.step()
        v.append(deepcopy(IN.v))
    plt.plot(v)
    plt.show()







