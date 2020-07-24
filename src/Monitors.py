from copy import deepcopy
import numpy as np


class SpikeMonitor():
    '''
    runs withing the Network class and collects indices of neurons at which the spike has just occurred
    '''
    def __init__(self, Neurons):
        self.N = len(Neurons)
        # spike times will be a dictionary with neural indices as keys and spike times as entries
        self.spike_times = dict()
        for i in range(self.N):
            self.spike_times[i] = []

    def update(self, spike_list, current_time):
        # spike list is the list of neurons which have spiked at this time step
        for ind in spike_list:
            self.spike_times[ind].append(current_time)
        return None


class StateMonitor():
    '''
    runs withing the Network class and collects the specified variables from the neurons
    '''
    def __init__(self, Neurons, variables_to_monitor):
        self.N = len(Neurons)
        self.Neurons = Neurons # make sure it is a reference not an entirely new object! (seems ok though)
        self.t_range = np.array([0.0])
        self.variables_to_monitor = variables_to_monitor

        for var in self.variables_to_monitor:
            exec(f"self.{var} = np.empty((0, self.N), dtype=float)")

    def update_history(self, current_time):
        self.t_range = np.append(self.t_range, current_time)

        for var in self.variables_to_monitor:
            # collect variables from neurons

            vals = []
            for i in range(len(self.Neurons)):
                vals.append(eval(f"self.Neurons[{i}].{var}"))

            tmp = deepcopy(np.array([vals]))
            # append the relevant variable to the statemonitor history
            exec(f"self.{var} = np.append(self.{var}, tmp, axis = 0)")
        return None

class SynapseMonitor():
    def __init__(self, synapses):
        self.synapses = synapses # make sure it is a reference not an entirely new object! (seems ok though)
        self.t_range = np.array([0.0])
        self.syn_weights = dict()
        for key in (list(self.synapses.keys())):
            self.syn_weights[key] = []

    def update_history(self, current_time):
        self.t_range = np.append(self.t_range, current_time)

        for key in (list(self.synapses.keys())):
            self.syn_weights[key].append(deepcopy(self.synapses[key].w))

        return None


