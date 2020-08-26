import matplotlib.pyplot as plt


def plot_nodal_demand(d):
    x_ = [0.5*i for i in range(d.shape[1])]
    for i in range(d.shape[0]):
        plt.plot(x_, d[i], label=str(i))
    plt.title("NZ Loads for 19 nodes, for september 2nd 2019 ")
    plt.xlabel("Time (hour)")
    plt.ylabel("Load (MWh)")
    plt.legend()
    plt.show()


def plot_lambdas_gammas(lambdas, gammas):
    fig, axs = plt.subplots(2, sharex=True, figsize=(15, 10), dpi=80, facecolor='w', edgecolor='k')
    plt.subplots_adjust(hspace=.0)
    fs = 20

    axs[0].plot(gammas)
    axs[0].set_ylabel('Average price $\gamma$ [\$/MW]', fontsize=fs)
    axs[0].set_ylim(bottom=0)
    axs[0].set_title('Average price and congestion curves\n Baseline model, Sept. 2nd, 2019', fontsize=fs)
    axs[0].grid()

    for i in range(1, 20):
        axs[1].plot(lambdas[i] - gammas, label=f'{i}')

    axs[1].legend()
    axs[1].set_ylabel('$\lambda - \gamma$ [\$/MW]', fontsize=fs)
    axs[1].set_xlabel('Time [h]', fontsize=fs)
    plt.xticks(range(0, 48, 2), range(24))
    axs[1].set_xlim([0, 48])
    axs[1].grid()
    plt.show()