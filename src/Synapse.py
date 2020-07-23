import numpy as np

class SynapseSTDP():
    def __init__(self, pre, post, dt=0.1):
        self.dt = dt
        self.pre = pre # neuron instance (presynaptic)
        self.post = post # neuron instance (postsynaptic)
        self.w = 0.1 * np.random.randn()
        self.h = self.w
        self.x = 0 # trace associated with presynaptic neuron spiking
        self.y = 0 # trace associated with postsynaptic neuron spiking
        self.A_plus = 0.04 # positive change if presynaptic neuron spikes
        self.A_minus = -0.04 # negative change if postsynaptic neuron spikes
        self.tau_plus = 10 # ms
        self.tau_minus = 10  # ms


    def update(self, pre_nrn, post_nrn):
        # update traces
        self.x = self.x - self.dt * (self.x / self.tau_plus)
        self.y = self.y - self.dt * (self.y / self.tau_minus)

        pre_spike = True if pre_nrn.spike_occurred else False
        post_spike = True if post_nrn.spike_occurred else False

        # update the postsynaptic voltage (immediate jump):
        if pre_spike:
            post_nrn.v += 40 * self.w

        # the learning part
        dh = 0
        # if a presynaptic neuron has spiked
        if pre_spike:
            self.x += self.A_plus
            dh += self.y

        # if a postsynaptic neuron has spiked
        if post_spike:
            self.y += self.A_minus
            dh += self.x

        # interestingly, if the both spike arrive at the same time, the change will be mitigated
        self.h += dh
        self.w = np.tanh(self.h)
        return None


    
