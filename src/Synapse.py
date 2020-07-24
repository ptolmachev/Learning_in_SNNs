from copy import deepcopy

import numpy as np

class SynapseSTDP():
    def __init__(self, pre, post, dt=0.1):
        # parameters in Spatial Properties of STDP in a
        # Self-Learning Spiking Neural
        # Network Enable Controlling a Mobile
        # Robot

        self.dt = dt
        self.pre = pre # neuron instance (presynaptic)
        self.post = post # neuron instance (postsynaptic)
        self.w = 0.1 + 0.3 * np.random.rand()
        self.h = self.w
        self.x = 1 # trace associated with presynaptic neuron spiking
        self.y = -1 # trace associated with postsynaptic neuron spiking
        self.A_plus =  1 # positive change if presynaptic neuron spikes
        self.A_minus = -1 # negative change if postsynaptic neuron spikes
        self.tau_plus = 10 # ms
        self.tau_minus = 10  # ms
        self.etha = 20
        self.alpha = 1
        self.lmbda = 0.01

    def propagate(self,pre_nrn, post_nrn):
        pre_spike = True if pre_nrn.spike_occurred else False
        # update the postsynaptic voltage (immediate jump):
        if pre_spike:
            post_nrn.v += self.etha * self.w
        return None

    def learn(self, pre_nrn, post_nrn):
        # update traces
        self.x = self.x - self.dt * (self.x / self.tau_plus)
        self.y = self.y - self.dt * (self.y / self.tau_minus)

        pre_spike = True if pre_nrn.spike_occurred else False
        post_spike = True if post_nrn.spike_occurred else False

        # the learning part
        dw = 0
        # if a presynaptic neuron has spiked
        if pre_spike:
            self.x += self.A_plus
            dw += self.lmbda * self.y * (1.0 - self.w)

        # if a postsynaptic neuron has spiked
        if post_spike:
            self.y += self.A_minus
            dw += self.lmbda * self.alpha * self.x * self.w # always positive if w > 0

        self.w = deepcopy(self.w + dw)
        return None