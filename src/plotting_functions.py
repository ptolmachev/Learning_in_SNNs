from matplotlib import pyplot as plt

def plot_synpatic_evolution(net, synapses):
    fig = plt.figure()
    syn_weights = net.synapmon.syn_weights
    t_range = net.statemon.t_range[1:]
    for key in synapses:
        # if key in net.plastic_synapses:
        plt.plot(t_range, syn_weights[key])
    return fig

def plot_spikes(net):
    spikes = net.spikemon.spike_times
    colors = ['r', 'g', 'b', 'm', 'k']
    fig = plt.figure()
    for i in range(net.N):
        plt.eventplot(spikes[i], lineoffsets=i, colors=colors[i % len(colors)])
    plt.ylim([-1, net.N])
    return fig

def plot_traces(net):
    v = net.statemon.v
    t_range = net.statemon.t_range[1:]
    fig = plt.figure()
    plt.plot(t_range, v)
    return fig

def plot_spikes_and_weights(net):
    spikes = net.spikemon.spike_times
    colors = ['r', 'g', 'b', 'm', 'k']
    fig, axes = plt.subplots(2, 1)
    for i in range(net.N):
        axes[0].eventplot(spikes[i], lineoffsets=i, colors=colors[i % len(colors)])
    axes[0].set_ylim([-1, net.N])

    syn_weights = net.synapmon.syn_weights
    t_range = net.statemon.t_range[1:]
    for key in list(syn_weights.keys()):
        if key in net.plastic_synapses:
            axes[1].plot(t_range, syn_weights[key])

    axes[0].set_xlim([0, t_range[-1]])
    axes[1].set_xlim([0, t_range[-1]])
    return fig
