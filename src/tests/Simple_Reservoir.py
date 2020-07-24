from src.Connections import connect_reservoir
from src.Network import Network
from src.Neuron import LIFNeuron, IzhikevichNeuron
from src.SpikeTrain import UniformSpikeTrain
from src.plotting_functions import *
from matplotlib import pyplot as plt
import numpy as np

N = 30  # number of neurons
dt = 0.1  # granularity of time
T = 1000  # ms to run the network for
T_steps = int(T / dt)  # time steps to run the network for
variables_to_monitor = ['v']

neurons = [IzhikevichNeuron() for i in range(N)]

enf_nrns = [0, 1, 2, N - 1] # the list of indices of the enforced neurons
synapses = connect_reservoir(inps=enf_nrns[:3], outs=enf_nrns[3:], N=N, prob=0.5)
net = Network(dt, neurons, synapses, variables_to_monitor)
input_spike_train = UniformSpikeTrain(T).generate(15) # in Hz
output_spike_train = UniformSpikeTrain(T).generate(30)

net.enforce_neurons(enf_nrns, [input_spike_train, input_spike_train + 60, input_spike_train - 47, output_spike_train])

# connections from all the neurons to the output neuron
plastic_synapses = []
for i in range(N - 1):
    plastic_synapses.append((i, N - 1))

net.set_plastic_synapses(plastic_synapses)
net.run(T_steps)

# # Plots
fig1 = plot_traces(net)
plt.show(block = True)
plt.close()

fig2 = plot_spikes(net)
plt.show(block = True)
plt.close()

fig3 = plot_synpatic_evolution(net)
plt.show(block=True)
plt.close()