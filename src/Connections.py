from src.Synapse import Synapse
import numpy as np

def connect_reservoir(neurons, inps, outs, prob):
    N = len(neurons)
    synapses = dict()
    # for the neurons in the bulk of the Reservoir
    for i in range(N):
        for j in range(N):
            if not ((i in inps) or (i in outs) or (j in inps) or (j in outs)):
                if (i != j) and (np.random.rand() <= prob):
                    synapses[i, j] = Synapse(pre=neurons[i], post=neurons[j])

    # connect input neurons to all the neurons in the network
    for i in inps:
        for j in range(N):
            if not (j in inps):
                synapses[i, j] = Synapse(pre=neurons[i], post=neurons[j])

    # connect all the neurons in the network to the outputs (apart from inputs and the other outputs)
    for j in outs:
        for i in range(N):
            if not (i in outs):
                synapses[i, j] = Synapse(pre=neurons[i], post=neurons[j])
    return synapses