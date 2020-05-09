import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

d_t_df = pd.read_csv('data/results/test3/df_demand.csv')
d_t_df = d_t_df.rename(columns={'Unnamed: 0': "t"})

gamma_df = pd.read_csv('data/results/test3/df_gamma.csv')
gamma_df = gamma_df.rename(columns={'Unnamed: 0': "t", "0" : "gamma"})

LMP_df = pd.read_csv('data/results/test3/df_lambda.csv')
LMP_df = LMP_df.rename(columns={'Unnamed: 0': "Node"})

p_t_df = pd.read_csv('data/results/test3/df_p_t.csv')
p_t_df = p_t_df.rename(columns={'Unnamed: 0': "Node"})

z_df = pd.read_csv('data/results/test3/df_z.csv')
z_df = z_df.rename(columns={'Unnamed: 0': "t"})

Network = pd.read_csv('data/ABM/ABM_Network_details.csv')
SimpNetwork = pd.read_csv('data/ABM/ABM_Simplified_network.csv')
SimpNetwork.rename(columns = {
    'Swem Node' : 'SimpNode',
    ' NZEM Substations that act as Grid Exit Points' : 'OriginNodes'
}, inplace=True)
DictSimpNetwork = {
        snode: list(set([onode[:3]
                         for onode in SimpNetwork.OriginNodes[SimpNetwork.SimpNode == snode]
                        .values[0]
                        .split(' ')[1:]]))
    for snode in SimpNetwork.SimpNode
    }

Nodes = np.unique(np.concatenate((np.unique(Network.LEAVE), np.unique(Network.ENTER))))
Nodes[0], Nodes[1] = Nodes[1], Nodes[0]
m = Network.shape[0]
Network['NLeave'] = np.array([np.where(Nodes == Network['LEAVE'][l])[0][0] for l in range(m)])
Network['NEnter'] = np.array([np.where(Nodes == Network['ENTER'][l])[0][0] for l in range(m)])


def plot_congestion(Nodes, gamma_df, LMP_df, t, Node = None):
    # Setting up the data to plot
    fs = 15
    if Node is not None:
        Node_name = Nodes[Node]
        Nodes = [None]*len(Nodes)
        Nodes[Node] = Node_name
        Y = np.array(LMP_df[[f'{i}' for i in range(t)]]).T - np.array([gamma_df.gamma[:t].tolist()]).T

    else:
        Y = np.array(LMP_df[[f'{i}' for i in range(t)]]).T - np.array([gamma_df.gamma[:t].tolist()]).T

    plt.figure(num=1, figsize=(15, 8), dpi=80, facecolor='w', edgecolor='k')
    lines = plt.plot(Y)
    plt.legend(lines, list(Nodes), fontsize = fs)
    plt.ylabel('$\lambda - \gamma$ [\$/MW]', fontsize = fs)
    plt.xlabel('Time t [30 min]', fontsize = fs)
    plt.xlim([0,48])
    plt.title('Congestion curves of the NZ network \n 1st week of september 2019', fontsize = fs)
    plt.show()

def plot_profit(z_df, LMP_df, Node, t):
    # Setting up the data to plot
    fs = 15
    u = np.array([z_df.u[:t]]).T
    LMP = np.array(LMP_df[[f'{i}' for i in range(t)]][LMP_df.Node == Node]).T
    Y = u*LMP
    Y_cum = np.cumsum(Y)

    plt.figure(num=1, figsize=(15, 8), dpi=80, facecolor='w', edgecolor='k')
    plt.plot(Y, label = 'Gains')
    plt.plot(Y_cum, label = 'Profit')
    plt.axhline(y = 0, color = 'k', ls = '--')
    plt.ylabel('Profit [\$]', fontsize=fs)
    plt.xlabel('Time t [30 min]', fontsize=fs)
    plt.xlim([0, 48])
    plt.legend(fontsize = fs)
    plt.title(f'Profit from battery placement \n Node = {Node}', fontsize = fs)
    plt.show()


def plot_congestion_profit(Nodes, gamma_df, z_df, LMP_df, Node, t):
    fig, axs = plt.subplots(2, sharex = True, figsize=(15, 8), dpi=80, facecolor='w', edgecolor='k')
    plt.subplots_adjust(hspace=.0)
    fs = 20

    # Profits
    u = np.array([z_df.u[:t]]).T
    LMP = np.array(LMP_df[[f'{i}' for i in range(t)]][LMP_df.Node == Node]).T
    Y = u * LMP
    Y_cum = np.cumsum(Y)

    axs[0].plot(Y, label='Gains')
    axs[0].plot(Y_cum, label='Profit')
    axs[0].axhline(y=0, color='k', ls='--')
    axs[0].set_ylabel('Profit [\$]', fontsize=fs)
    axs[0].set_title(f'Profit and congestions, 1st week of september 2019 \n Node = {Node}, {Nodes[Node]}', fontsize=fs)
    axs[0].legend(fontsize=fs)

    # Congestion

    Y = np.array(LMP_df[[f'{i}' for i in range(t)]]).T - np.array([gamma_df.gamma[:t].tolist()]).T
    for i in range(len(Nodes)):
        if i == 10:
            axs[1].plot(Y[:,i], label = Nodes[i])
        else:
            axs[1].plot(Y[:,i])
    axs[1].legend(fontsize = fs)
    axs[1].set_ylabel('$\lambda - \gamma$ [\$/MW]', fontsize=fs)
    axs[1].set_xlabel('Time t [h]', fontsize=fs)
    axs[1].set_xlim([0, 48])

plt.figure()
plt.plot(np.array(d_t_df[[f'{i}' for i in range(48)]]).T)
plt.xlabel('Time [h]', fontsize = 15)
plt.ylabel('Demand [MW]', fontsize = 15)
plt.title('Demand curves', fontsize = 15)
plt.show()


"""
TESTS
"""

plot_congestion(Nodes, gamma_df, LMP_df, 48, Node = 2)
plot_profit(z_df, LMP_df, 10, 48)
plot_congestion_profit(Nodes, gamma_df, z_df, LMP_df, 10, 18)