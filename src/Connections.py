from src.Synapse import SynapseSTDP
import numpy as np

def connect_reservoir(inps, outs, N, prob):
    synapses = dict()
    # for the neurons in the bulk of the Reservoir
    for i in range(N):
        for j in range(N):
            if (i != j) and not ((i in inps) or (j in outs)):
                if np.random.rand() <= prob:
                    synapses[i, j] = SynapseSTDP(pre=i, post=j)

    # connect input neurons to all the neurons in the network (apart from other inputs and the outputs)
    for i in inps:
        for j in range(N):
            if not ((i in inps) or (j in outs)):
                synapses[i, j] = SynapseSTDP(pre=i, post=j)

    # connect all the neurons in the network to the outputs (apart from inputs and the other outputs)
    for j in outs:
        for i in range(N):
            if not ((i in inps) or (i in outs)):
                synapses[i, j] = SynapseSTDP(pre=i, post=j)
    return synapses