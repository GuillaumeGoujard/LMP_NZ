from lmpnz.Network.Topology.add_arrow import add_arrow
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import random
from lmpnz.Network.Topology.Topology import create_incidence
from lmpnz.Network.Topology.Topology import create_adjacency
from lmpnz.Network.Topology.Topology import create_H_hat


# Files
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

Sites = pd.read_csv('data/topology/Sites.csv')
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

# Order of nodes used for H and solver

m = Network.shape[0]

Nodes = np.unique(np.concatenate((np.unique(Network.LEAVE), np.unique(Network.ENTER))))
Nodes[0], Nodes[1] = Nodes[1], Nodes[0]

Network['NLeave'] = np.array([np.where(Nodes == Network['LEAVE'][l])[0][0] for l in range(m)])
Network['NEnter'] = np.array([np.where(Nodes == Network['ENTER'][l])[0][0] for l in range(m)])

I = create_incidence(Network.NLeave, Network.NEnter)
A = create_adjacency(Network.NLeave, Network.NEnter)

omega_NZ = 50*(2*np.pi)
z = Network['Resistance (Ohms)'] + 1j*Network["Reactance (Ohms)"]*omega_NZ
y = 1/z
y = np.imag(y)
H_hat = create_H_hat(I, y)

locations = Sites.MXLOCATION.unique().tolist()
locations.remove('WKM')
locations.remove('HLY')
def X_node(node):
    if node in locations:
        x_value = Sites.X[Sites.MXLOCATION == node].values[0]
    elif node == 'WKM':
        x_value = (Sites.X[Sites.MXLOCATION == node] + 50000).values[0]
    elif node == 'B_star':
        node1 = Sites.X[Sites.MXLOCATION == 'HAY'].values[0]
        node2 = Sites.X[Sites.MXLOCATION == 'TWZ'].values[0]
        x_value = node1 + 0.25*(node2 - node1)
    else:
        x_value = Sites.X[Sites.MXLOCATION.apply(lambda x: x in DictSimpNetwork[node])].mean()

    return x_value
def Y_node(node):
    if node in locations:
        y_value = Sites.Y[Sites.MXLOCATION == node].values[0]
    elif node == 'WKM':
        y_value = Sites.Y[Sites.MXLOCATION == node].values[0] + 50000
    elif node == 'B_star':
        node1 = Sites.Y[Sites.MXLOCATION == 'HAY'].values[0]
        node2 = Sites.Y[Sites.MXLOCATION == 'TWZ'].values[0]
        y_value = node1 + 0.25 * (node2 - node1)
    else:
        y_value = Sites.Y[Sites.MXLOCATION.apply(lambda x: x in DictSimpNetwork[node])].mean()

    return y_value

X = pd.Series(Nodes).apply(lambda node: X_node(node)).tolist()
Y = pd.Series(Nodes).apply(lambda node: Y_node(node)).tolist()

f = 10
Nodes_df = pd.DataFrame({
    'Node': Nodes,
    'x': X,
    'y': Y,
    'd_t': [0]*20,
    'g_t': [0]*20,
    'p_t': [0]*20,
    'u_t': [0]*20
})

Node_leave = Network.LEAVE.tolist()
x1 = pd.Series(Node_leave).apply(lambda node: X_node(node)).tolist()
y1 = pd.Series(Node_leave).apply(lambda node: Y_node(node)).tolist()
Node_enter = Network.ENTER.tolist()
x2 = pd.Series(Node_enter).apply(lambda node: X_node(node)).tolist()
y2 = pd.Series(Node_enter).apply(lambda node: Y_node(node)).tolist()

Lines_df = pd.DataFrame({
    'Node_leave' : Node_leave,
    'x1' : x1,
    'y1' : y1,
    'Node_enter' : Network.ENTER.tolist(),
    'x2' : x2,
    'y2' : y2,
    'f' : [0]*m,
    'normed_f' : [0]*m
})


u_PM = z_df.u.values
u_PT = z_df.u.values + 1



def plot_pricemaker_pricetaker_profits(u_PT, u_PM, LMP_df, node_PT, node_PM, Nodes):
    '''

    :param u_PT: (T,1) vector, np.array
    :param u_PM: (T,1) vector, np.array
    :param LMP_df: (19,T) matrix, np.array
    :param node_PT: integer
    :param node_PM: integer
    :param Nodes: list of nodes (at each index, the name of the corresponding node)
    :return: fig, ax
    '''

    fig, ax = plt.subplots(figsize=(15, 8), dpi=80, facecolor='w', edgecolor='k')
    fs = 15

    node_PM_name = Nodes[node_PM]
    node_PT_name = Nodes[node_PT]

    # Plotting PM/PT injection at their optimized node as lineplot
    ax.plot(u_PM, label = f'PM battery injection at node {node_PM_name}')
    ax.plot(u_PT, label = f'PT battery injection at node {node_PT_name}')
    ax.set_ylabel('Injection [in MWh]', fontsize=fs)
    ax.set_xlabel('Time t [30 min]', fontsize=fs)
    ax.legend(fontsize=fs, loc = 'upper left')

    # Calculating profit at each timestep
    profit_PM = u_PM * LMP_df.values[node_PM,1:]
    profit_PT = u_PT * LMP_df.values[node_PT,1:]

    # Plotting profit at each timestep as barplot
    ax2 = ax.twinx()
    ax2.bar(x = list(range(48)), height = profit_PM, width =  0.2, align = 'edge',label = f'PM battery profit at node {node_PM_name}')
    ax2.bar(x = list(range(48)), height = profit_PT, width = -0.2, align = 'edge',label = f'PT battery profit at node {node_PT_name}')
    ax2.set_ylabel('Profit (in \$)', color="blue", fontsize=fs)
    ax2.legend(fontsize=fs, loc = 'upper right')

    ax.set_xlabel('Time t [30 min]', fontsize=fs)

    ax.set_title(f'Injection and Profit from battery placement \n '
                 f'Price Maker Node = {node_PM_name} \n '
                 f'Price Taker Node = {node_PT_name}', fontsize=fs)
    return fig


fig = plot_pricemaker_pricetaker_profits(u_PT, u_PM, LMP_df, 10, 12, Nodes)
plt.show()
