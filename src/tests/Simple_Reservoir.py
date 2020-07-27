from copy import deepcopy

from src.Connections import connect_reservoir
from src.Network import Network
from src.Neuron import LIFNeuron, IzhikevichNeuron
from src.SpikeTrain import UniformSpikeTrain, PoissonSpikeTrain
from src.plotting_functions import *
from matplotlib import pyplot as plt
import numpy as np

#TODO: put the network to "the edge of chaos"
N = 30  # number of neurons
dt = 0.1  # granularity of time
T = 12000  # ms to run the network for
T_steps = int(T / dt)  # time steps to run the network for
variables_to_monitor = [] #['v']

neurons = [IzhikevichNeuron() for i in range(N)]
enf_nrns = [0, 1, 2, N - 1] # the list of indices of the enforced neurons
synapses = connect_reservoir(neurons, inps=enf_nrns[:3], outs=enf_nrns[3:], prob=0.3)
net = Network(dt, neurons, synapses, variables_to_monitor)

# generate template input and output spike trains which repeats every T/10 timesteps
ist1= PoissonSpikeTrain(T / 10).generate(11) # in Hz
ist2= PoissonSpikeTrain(T / 10).generate(15)
ist3= PoissonSpikeTrain(T / 10).generate(13)
ost = PoissonSpikeTrain(T / 10).generate(14)
input_spike_train1 = np.hstack([ist1 + i * (T / 10) for i in range(20)])
input_spike_train2 = np.hstack([ist2 + i * (T / 10) for i in range(20)])
input_spike_train3 = np.hstack([ist3 + i * (T / 10) for i in range(20)])
output_spike_train = np.hstack([ost + i * (T / 10) for i in range(20)])

# fix the evolution of specified neurons and force them to spike at with the specified patterns
net.enforce_neurons(enf_nrns, [input_spike_train1, input_spike_train2, input_spike_train3, output_spike_train])

# make the connections from all the neurons to the output neuron plastic (able to learn)
plastic_synapses = []
for i in range(N - 1):
    plastic_synapses.append((i, N - 1))
plot_synaptic_traces = deepcopy(plastic_synapses)
net.set_plastic_synapses(plastic_synapses)

net.run(int(T_steps*1.5))

# remove all the plasticity
net.set_plastic_synapses([])
enf_nrns = [0, 1, 2]
net.enforce_neurons(enf_nrns, [input_spike_train1, input_spike_train2, input_spike_train3])
net.run(int(T_steps*0.5))

# # # Plots
# fig1 = plot_traces(net)
# plt.show(block = True)
# plt.close()

fig2 = plot_spikes(net)
plt.show(block = True)
plt.close()

fig3 = plot_synpatic_evolution(net, plot_synaptic_traces)
plt.show(block=True)
plt.close()