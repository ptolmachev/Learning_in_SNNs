from src.Network import Network
from src.Neuron import IzhikevichNeuron
from src.SpikeTrain import UniformSpikeTrain, PoissonSpikeTrain
from src.Synapse import SynapseSTDP
from src.plotting_functions import *
from matplotlib import pyplot as plt

neurons = [IzhikevichNeuron() for i in range(2)]
dt = 0.1
T = 20000  # ms to run the network for
T_steps = int(T / dt)  # time steps to run the network for
variables_to_monitor = ['v']
synapses = dict()
synapses[0, 1] = SynapseSTDP(neurons[0], neurons[1])

net = Network(dt, neurons, synapses, variables_to_monitor)
net.set_plastic_synapses(list_of_tuples = [(0, 1)])
spike_train_uniform = UniformSpikeTrain(T).generate(10)
spike_train_poisson = PoissonSpikeTrain(T).generate(10)
net.enforce_neurons(indices=[0, 1], spike_trains=[spike_train_uniform, spike_train_poisson])
# net.enforce_neurons(indices=[0], spike_trains=[spike_train_uniform])

net.run(T_steps)

fig2 = plot_spikes_and_weights(net)
plt.show(block = True)