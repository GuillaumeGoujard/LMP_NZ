import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

d_t_df = pd.read_csv('data/results/test5/df_demand.csv')
d_t_df = d_t_df.rename(columns={'Unnamed: 0': "t"})

gamma_df = pd.read_csv('data/results/test5/df_gamma.csv')
gamma_df = gamma_df.rename(columns={'Unnamed: 0': "t", "0" : "gamma"})

LMP_df = pd.read_csv('data/results/test5/df_lambda.csv')
LMP_df = LMP_df.rename(columns={'Unnamed: 0': "Node"})

p_t_df = pd.read_csv('data/results/test5/df_p_t.csv')
p_t_df = p_t_df.rename(columns={'Unnamed: 0': "Node"})

z_df = pd.read_csv('data/results/test5/df_z.csv')
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
    fig, axs = plt.subplots(2, sharex=True, figsize=(15, 10), dpi=80, facecolor='w', edgecolor='k')
    plt.subplots_adjust(hspace=.0)
    fs = 20

    if Node is not None:
        Node_name = Nodes[Node]
        Nodes = [None]*len(Nodes)
        Nodes[Node] = Node_name
        Y = np.array(LMP_df[[f'{i}' for i in range(t)]][1:]).T - np.array([gamma_df.gamma[:t].tolist()]).T
    else:
        Y = np.array(LMP_df[[f'{i}' for i in range(t)]][1:]).T - np.array([gamma_df.gamma[:t].tolist()]).T

    axs[0].plot(gamma[0])
    axs[0].set_ylabel('Average price $\gamma$ [\$/MW]', fontsize = fs)
    axs[0].set_ylim(bottom = 0)
    axs[0].set_title('Average price and congestion curves\n Baseline model, Sept. 2nd, 2019', fontsize = fs)
    axs[0].grid()


    for i, y_arr, label in zip(range(1, 20), Y[1:,:], Nodes[1:].tolist()):
        if (label == 'MDN') | (label == 'HEN'):
            axs[1].plot(y_arr, label=f'{i} : {label}', linewidth=5)
        else:
            axs[1].plot(y_arr, label=f'{i} : {label}')

    axs[1].legend()
    axs[1].set_ylabel('$\lambda - \gamma$ [\$/MW]', fontsize = fs)
    axs[1].set_xlabel('Time [h]', fontsize = fs)
    plt.xticks(range(0,48,2), range(24))
    axs[1].set_xlim([0,48])
    axs[1].grid()
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


# plt.figure(num = 2, figsize=(15, 8), dpi=80, facecolor='w', edgecolor='k')
# for i, y_arr, label in zip(range(1,20), np.array(d_t_df[[f'{i}' for i in range(48)]][1:]).tolist(), Nodes[1:].tolist()):
#     if (label == 'MDN') | (label =='HEN'):
#         plt.plot(y_arr, label = f'{i} : {label}', linewidth = 5)
#     else:
#         plt.plot(y_arr, label=f'{i} : {label}')
# plt.xlabel('Time [h]', fontsize = 15)
# plt.ylabel('Demand [MW]', fontsize = 15)
# plt.title('Load curves\n Sept. 2nd, 2019', fontsize = 15)
# plt.xticks(range(0,48,2), range(24))
# plt.legend(fontsize = 15)
# plt.show()


# Profits and SOC at node 10
u = np.array([z_df.u]).T
LMP = np.array(LMP_df[[f'{i}' for i in range(48)]][LMP_df.Node == 10]).T
Y = u * LMP

fs = 15
fig, ax = plt.subplots(figsize=(15, 8), dpi=80, facecolor='w', edgecolor='k')
ax1 = ax.twinx()

ax.plot(range(48), Y, color = 'g')
ax.set_ylabel('Profit [\$]', fontsize = fs, color = 'g')
ax.set_xlabel('Timestep [h]', fontsize = fs)
ax.set_ylim([-6000,6000])

ax1.plot(range(48), z_df.z, color = 'b')
ax1.set_ylabel('SOC', fontsize = fs, color = 'b')
ax1.set_ylim([-70,70])

plt.setp(ax, xticks = range(0,48,2), xticklabels = range(24))
plt.title('PM profits and SOC\n 5 year battery, Sept. 2nd, 2019', fontsize = fs)
plt.grid()



# Profits and SOC at node 10

# fs = 15
# fig, ax = plt.subplots(figsize=(15, 8), dpi=80, facecolor='w', edgecolor='k')
# ax1 = ax.twinx()
#
# ax.plot(range(48), Y, color = 'g')
# ax.set_ylabel('Profit [\$]', fontsize = fs, color = 'g')
# ax.set_xlabel('Timestep [h]', fontsize = fs)
# ax.set_ylim([-6000,6000])
#
# ax1.plot(range(48), z_df.z, color = 'b')
# ax1.set_ylabel('SOC', fontsize = fs, color = 'b')
# ax1.set_ylim([-70,70])
#
# plt.setp(ax, xticks = range(0,48,2), xticklabels = range(24))
# plt.title('PM profits and SOC\n 5 year battery, Sept. 2nd, 2019', fontsize = fs)
# plt.grid()







"""
TESTS
"""

plot_congestion(Nodes, gamma_df, LMP_df, 48)
# plot_profit(z_df, LMP_df, 10, 48)
# plot_congestion_profit(Nodes, gamma_df, z_df, LMP_df, 10, 18)








# Cumulated profit for PM and PT
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