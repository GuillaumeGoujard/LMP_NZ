from main.Network.Topology.add_arrow import add_arrow
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import random
from main.Network.Topology.Topology import create_incidence
from main.Network.Topology.Topology import create_adjacency
from main.Network.Topology.Topology import create_H
import stored_path

import networkx as nx

# Files
Sites = pd.read_csv(stored_path.main_path+'/data/topology/Sites.csv')
Network = pd.read_csv(stored_path.main_path+'/data/ABM/ABM_Network_details.csv')
SimpNetwork = pd.read_csv(stored_path.main_path+'/data/ABM/ABM_Simplified_network.csv')

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
H = create_H(I,y)
H_hat = H[:H.shape[0]//2,:]
# H_hat = create_H_hat(I, y)

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

def plot_NZ_market_offer_demand(Nodes_df,Lines_df,d_t,g_t):

    Nodes_df.d_t = d_t
    Nodes_df.g_t = g_t

    fig = plt.figure()
    ax = fig.add_subplot(111)
    # Dots
    f = 100

    ax.scatter(Nodes_df.x.values,
               Nodes_df.y.values,
               s=f*Nodes_df.d_t.values/Nodes_df.d_t.max(),
               c='g',
               alpha=0.2,
               label='Demand')
    ax.scatter(Nodes_df.x.values,
               Nodes_df.y.values,
               s=f*Nodes_df.g_t.values/Nodes_df.g_t.max(),
               c='b',
               alpha=0.2,
               label='Supply')

    Lines_df['Line2D'] = Lines_df[['x1', 'x2', 'y1', 'y2']].apply(
        lambda row: Line2D(*[(row[0], row[1]), (row[2], row[3])], alpha=1), axis=1)
    for i, line in enumerate(Lines_df['Line2D'].values.tolist()):
        ax.add_line(line)

    for i, node in enumerate(Nodes_df.Node.values):
        ax.annotate(f'{node}: \n d = {round(Nodes_df.d_t.values[i])} MW \n g = {round(Nodes_df.g_t.values[i])} MW',
                    (Nodes_df[Nodes_df.Node == node].x.values[0],
                     Nodes_df[Nodes_df.Node == node].y.values[0] + 10000),
                    fontsize=15)

    ax.set_xlabel('Longitude (X)')
    ax.set_ylabel('Latitude (Y)')
    ax.axis('equal')
    ax.legend()

def plot_NZ_market_clearing(Nodes_df, Lines_df, H_hat, p_t, u_t):
    Nodes_df.p_t = p_t
    Nodes_df.u_t = u_t

    Lines_df.f = H_hat@np.array([p_t]).T
    Lines_df.normed_f = Lines_df.f/Lines_df.f.abs().max()

    fig = plt.figure()
    ax = fig.add_subplot(111)

    f = 5000

    ax.scatter(Nodes_df.x.values,
               Nodes_df.y.values,
               s=f*np.abs(Nodes_df.p_t.values/np.abs(Nodes_df.p_t).max()),
               c='r',
               alpha=0.2,
               label='Injection')

    ax.scatter(Nodes_df.x.values,
               Nodes_df.y.values,
               s=Nodes_df.u_t.values,
               c='k',
               marker='+',
               label='Optimal battery placement')

    Lines_df['Arrow'] = Lines_df.f.apply(np.sign)
    Lines_df['Line2D'] = Lines_df[['x1', 'x2', 'y1', 'y2', 'normed_f']].apply(
        lambda row: Line2D(*[(row[0], row[1]), (row[2], row[3]), 15*row[4]], alpha=0.2), axis=1)
    for i, line in enumerate(Lines_df.Line2D.values.tolist()):
        ax.add_line(line)
        add_arrow(line, size=15*Lines_df.normed_f.values[i])

    for i, node in enumerate(Nodes_df.Node.values):
        ax.annotate(f'{node}: p = {round(Nodes_df.p_t.values[i])} MW',
                    (Nodes_df[Nodes_df.Node == node].x.values[0],
                     Nodes_df[Nodes_df.Node == node].y.values[0] + 10000),
                    fontsize=15)
    ax.set_xlabel('Longitude (X)')
    ax.set_ylabel('Latitude (Y)')
    ax.axis('equal')


d_t = [0] + [f*random.uniform(0,150) for i in range(19)]
Mng_t = [0] + [f*random.uniform(0,150) for i in range(19)]
p_t =[0] + [f*random.uniform(-150,150) for i in range(19)]
u_t = [0,1000] + [0]*18

plot_NZ_market_offer_demand(Nodes_df, Lines_df, d_t, g_t)
plt.show()
plot_NZ_market_clearing(Nodes_df, Lines_df, H_hat, p_t, u_t)
plt.show()