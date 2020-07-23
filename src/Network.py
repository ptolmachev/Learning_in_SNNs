import numpy as np
from src.Neuron import *
from src.Monitors import *
from src.Synapse import SynapseSTDP
from matplotlib import pyplot as plt

class Network():
    def __init__(self, dt, Neurons, Synapses, variables_to_monitor):
        # check if they have consistent dimensions
        self.N = len(Neurons) # number of neurons
        self.neurons = Neurons # a list of neurons
        self.synapses = Synapses # a dictionary with tuples (i, j) as keys and synapses as entries. i -> j

        self.dt = dt
        # set the same dt for every neuron and synapse
        self.set_dt()

        self.spikemon = SpikeMonitor(self.neurons)
        self.statemon = StateMonitor(self.neurons, variables_to_monitor = variables_to_monitor)
        self.synapmon = SynapseMonitor(self.synapses)
        self.t = 0


    def set_dt(self):
        for i in range(len(self.neurons)):
            self.neurons[i].dt = self.dt

        for key in list(self.synapses.keys()):
            self.synapses[key].dt = self.dt
        return None

    def update_neurons(self):
        for nrn in self.neurons:
            nrn.step()
        return None

    def update_spikemonitor(self):
        spike_list = []
        for i, nrn in enumerate(self.neurons):
            if nrn.spike_occurred:
                spike_list.append(i)
        self.spikemon.update(spike_list, self.t)
        return None

    def update_statemonitor(self):
        self.statemon.update_history(self.t)
        return None

    def update_synapmonitor(self):
        self.synapmon.update_history(self.t)
        return None

    def update_synapses(self):
        # for each synapse in the dictionary
        for key in list(self.synapses.keys()):
            pre_ind = key[0]
            post_ind = key[1]
            pre_nrn = self.neurons[pre_ind]
            post_nrn = self.neurons[post_ind]
            self.synapses[key].update(pre_nrn, post_nrn)
        return None


    def step(self):
        # update dt
        self.t += self.dt
        # update neurons
        self.update_neurons()
        # update synapses
        self.update_synapses()
        # update spikemonitor
        self.update_spikemonitor()
        self.update_statemonitor()
        self.update_synapmonitor()
        return None

    def run(self, T_steps):
        for i in range(T_steps):
            self.step()
        return None


if __name__ == '__main__':
    N = 16
    dt = 0.1
    T = 5000
    T_steps = int(T / dt)
    variables_to_monitor = ['v']

    neurons = [LIFNeuron() for i in range(N)]

    synapses = dict()
    for i in range(N):
        for j in range(N):
            if np.random.rand() <= 0.25 and (i != j):
                synapses[i,j] = SynapseSTDP(pre=i, post=j)

    net = Network(dt, neurons, synapses, variables_to_monitor)
    net.run(T_steps)

    v = net.statemon.v
    t_range = net.statemon.t_range[1:]
    fig1 = plt.figure()
    plt.plot(t_range, v)
    plt.show()

    spikes = net.spikemon.spike_times
    colors = ['r', 'g', 'b', 'm', 'k']
    fig2 = plt.figure()
    for i in range(N):
        plt.eventplot(spikes[i], lineoffsets=i, colors=colors[i % len(colors)])
    plt.ylim([-1, N])
    plt.show()

    fig3 = plt.figure()
    syn_weights = net.synapmon.syn_weights
    t_range = net.statemon.t_range[1:]
    for key in list(syn_weights.keys()):
        plt.plot(t_range, syn_weights[key])
    plt.show()







