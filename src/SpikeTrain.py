import numpy as np
from matplotlib import pyplot as plt

class SpikeTrainGenerator():
    def __init__(self, T_range):
        self.T_range = T_range


class PoissonSpikeTrain(SpikeTrainGenerator):
    def __init__(self, T_range):
        super().__init__(T_range)

    def generate(self, r):
        t = 0
        spike_train = [0]
        while t < self.T_range:
            u = np.random.rand()
            delta_t = -(np.log(1 - u)) / (r / 1000)
            t = spike_train[-1] + delta_t
            if t < self.T_range:
                spike_train.append(t)
        return np.array(spike_train[1:]) #discregard the first 0 fictive spike

class UniformSpikeTrain(SpikeTrainGenerator):
    def __init__(self, T_range):
        super().__init__(T_range)

    def generate(self, r):
        t = 0
        spike_train = [0]
        while t < self.T_range:
            delta_t = (1 / (r / 1000))
            t = spike_train[-1] + delta_t
            spike_train.append(t)
        return np.array(spike_train[1:]) #discregard the first 0 fictive spike

if __name__ == '__main__':
    st = PoissonSpikeTrain(500).generate(r=10)
    plt.eventplot(st)
    plt.show()

