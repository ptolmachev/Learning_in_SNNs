import numpy as np

from src.Connections import connect_reservoir
from src.Neuron import *
from src.Monitors import *
from src.SpikeTrain import UniformSpikeTrain
from src.Synapse import SynapseSTDP
from src.plotting_functions import *


class Network():
    def __init__(self, dt, Neurons, Synapses, variables_to_monitor):
        # TODO: check if Neurons and Synapses have consistent dimensions

        self.N = len(Neurons) # number of neurons
        self.neurons = Neurons # a list of neurons
        self.synapses = Synapses # a dictionary with tuples (i, j) as keys and synapses as entries. i -> j

        # set the same dt for every neuron and synapse
        self.dt = dt
        self.set_dt()

        self.spikemon = SpikeMonitor(self.neurons)
        self.statemon = StateMonitor(self.neurons, variables_to_monitor = variables_to_monitor)
        self.synapmon = SynapseMonitor(self.synapses)

        self.enforced_neurons = []
        self.plastic_synapses = []
        self.t = 0

    def set_dt(self):
        for i in range(len(self.neurons)):
            self.neurons[i].dt = self.dt
        for key in list(self.synapses.keys()):
            self.synapses[key].dt = self.dt
        return None

    def update_neurons(self):
        for i, nrn in enumerate(self.neurons):
            #branch for Reservoir neurons
            if not (i in self.enforced_neurons) :
                nrn.step()
            #branch for input and output neurons
            else:
                if i in self.enforced_neurons:
                    # determine if the network is at the time at which there should be a spike in any of the enforced neurons
                    ind = self.enforced_neurons.index(i)
                    t_diffs = np.abs(self.enforced_spike_trains[ind] - self.t)
                    spike_now = np.any(t_diffs <= 0.5 * self.dt)
                    self.neurons[i].spike_occurred = spike_now
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
            pre_ind, post_ind = key
            pre_nrn = self.neurons[pre_ind]
            post_nrn = self.neurons[post_ind]
            # hte effect of spiking in the presynaptic neuron is propagated to the postsynaptic neuron
            self.synapses[key].propagate(pre_nrn, post_nrn)

            if key in self.plastic_synapses:
                self.synapses[key].learn(pre_nrn, post_nrn)
        return None

    def enforce_neurons(self, indices, spike_trains):
        '''
        Freeze the evolution of the specified neurons and provide the pattern according to which theses neurons should spike
        :param indices: the indices of the neurons whose evolution will be frozen and the spiking output will be replaced
        :param spike_trains: list of numpy arrays with the specified spike-timing. The length of the list = length indices
        '''
        self.enforced_neurons = indices
        self.enforced_spike_trains = spike_trains
        for i in self.enforced_neurons:
            self.neurons[i].v = -np.inf # precautionary measure
        return None

    def set_plastic_synapses(self, list_of_tuples):
        s_keys = list(self.synapses.keys())
        for s in list_of_tuples:
            if s in s_keys:
                self.plastic_synapses.append(s)
        return None

    def step(self):
        self.t += self.dt
        self.update_neurons()
        self.update_synapses()
        self.update_spikemonitor()
        self.update_statemonitor()
        self.update_synapmonitor()
        return None

    def run(self, T_steps):
        for i in range(T_steps):
            self.step()
        return None










