from main.Network.Topology.add_arrow import add_arrow
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import networkx as nx

# Files
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
Nodes = np.unique(np.concatenate((np.unique(Network.LEAVE), np.unique(Network.ENTER))))
Nodes[0], Nodes[1] = Nodes[1], Nodes[0]

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


f = 5

Nodes_df = pd.DataFrame({
    'Node': Nodes,
    'x': X,
    'y': Y,
    'cum_demand': [0] + [10*f]*9 + [50*f]*10,
    'cum_supply': [0] + [150*f]*6 + [120*f]*5 + [210*f]*8,
    'p_t': [0] + [70*f]* 19,
    'u_t': [0,1] + [0]*18
})

# Plotting tests


fig = plt.figure()
ax = fig.add_subplot(111)
# Dots
ax.scatter(Nodes_df.x.values,
            Nodes_df.y.values,
            s = Nodes_df.cum_demand.values,
            c = 'g',
            alpha = 0.2)
ax.scatter(Nodes_df.x.values,
            Nodes_df.y.values,
            s = Nodes_df.cum_supply.values,
            c = 'r',
            alpha = 0.2)

# Annotations
for node in Nodes_df.Node.values:
    ax.annotate(node,
                 (Nodes_df[Nodes_df.Node == node].x.values[0],
                  Nodes_df[Nodes_df.Node == node].y.values[0]),
                 fontsize='large')
ax.set_xlabel('Longitude (X)')
ax.set_ylabel('Latitude (Y)')
ax.axis('equal')

Node_leave = Network.LEAVE.tolist()
x1 = pd.Series(Node_leave).apply(lambda node: X_node(node)).tolist()
y1 = pd.Series(Node_leave).apply(lambda node: Y_node(node)).tolist()
Node_enter = Network.ENTER.tolist()
x2 = pd.Series(Node_enter).apply(lambda node: X_node(node)).tolist()
y2 = pd.Series(Node_enter).apply(lambda node: Y_node(node)).tolist()

m = Network.shape[0]

Lines_df = pd.DataFrame({
    'Node_leave' : Node_leave,
    'x1' : x1,
    'y1' : y1,
    'Node_enter' : Network.ENTER.tolist(),
    'x2' : x2,
    'y2' : y2,
    'Power_flow' : [(-1)**i*(i+1) for i in range(m)]
})

Lines_df['Arrow'] = Lines_df.Power_flow.apply(np.sign)

Lines_df['Line2D'] = Lines_df[['x1','x2','y1','y2','Power_flow']].apply(lambda row: Line2D(*[(row[0],row[1]),(row[2],row[3]),np.abs(row[4])]), axis = 1)
for i, line in enumerate(Lines_df['Line2D'].values.tolist()):
    ax.add_line(line)
    add_arrow(line, size = Lines_df['Power_flow'].values[i]*3+10)

ax.scatter(Nodes_df.x.values,
            Nodes_df.y.values,
            color = 'k')


# plt.plot((x1,x2),(y1,y2), c = 'r')
# plt.xlabel('Longitude (X)')
# plt.ylabel('Latitude (Y)')
# plt.axis('equal')
# plt.show()


def plot_NZ_market_clearing(p_t,u_t):
    '''


    :return:
    plot figure
    '''





