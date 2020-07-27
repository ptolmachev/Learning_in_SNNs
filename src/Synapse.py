from copy import deepcopy
from collections import deque
import numpy as np

class Synapse():
    def __init__(self, pre, post, dt=0.1):
        # parameters in Spatial Properties of STDP in a
        # Self-Learning Spiking Neural
        # Network Enable Controlling a Mobile
        # Robot

        self.dt = dt
        self.pre = pre # neuron instance (presynaptic)
        self.post = post # neuron instance (postsynaptic)
        # make this synapse accessible from the pre and post neurons
        self.pre.outcoming_synapses.append(self)
        self.post.incoming_synapses.append(self)

        self.w = 0.1 + 0.3 * np.random.rand()
        self.h = self.w
        self.x = 0 # trace associated with presynaptic neuron spiking
        self.y = 0 # trace associated with postsynaptic neuron spiking
        self.A_pre = 1
        self.A_post = 1
        self.tau_pre = 10 # ms
        self.tau_post = 10  # ms
        self.etha = 20
        self.lmbda = 0.005
        self.delay = np.maximum(3 + 3 * np.random.randn(), 0) # synaptic delay in ms
        self.delay_steps = int(np.ceil(self.delay / self.dt + 1e-9))
        self.update_queue = deque(np.zeros(self.delay_steps).astype(bool), maxlen = self.delay_steps)


    def propagate(self,pre_nrn, post_nrn):
        pre_spike = True if pre_nrn.spike_occurred else False

        # append the spike into the waiting queue (because of the synaptic delay)
        self.update_queue.append(True) if pre_spike else self.update_queue.append(False)

        # update the postsynaptic voltage:
        if self.update_queue[0]:
            post_nrn.v += self.etha * self.w
        return None

    def rhs_w(self):
        # access the synapses conveying impulses to post neuron and calculate net input to the post neuron
        net_input = 0
        for synapse in self.post.incoming_synapses:
            net_input += synapse.w * synapse.x
        # dw_{ji}/dt = -(1 / 2) \lambda dE_{i}/dw_{ji} = -(1 / 2) \lambda d(y_{i} - \sum w_{pi} s_p) ** 2/ dw_{ji}
        res = self.lmbda * (self.y - net_input) * self.x
        return res

    def learn(self, pre_nrn, post_nrn):
        # update traces
        self.x = self.x - self.dt * (self.x / self.tau_pre)
        self.y = self.y - self.dt * (self.y / self.tau_post)

        pre_spike = True if pre_nrn.spike_occurred else False
        post_spike = True if post_nrn.spike_occurred else False

        # the learning part
        # if a presynaptic neuron has spiked
        if pre_spike:
            self.x += self.A_pre
        # if a postsynaptic neuron has spiked
        if post_spike:
            self.y += self.A_post

        self.w = self.w + self.dt * self.rhs_w()
        return None