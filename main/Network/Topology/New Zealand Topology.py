import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as lines
import numpy as np
import swifter

from main.Network.Topology.add_arrow import add_arrow

## Importing relevant csv files
Sites = pd.read_csv('data/topology/Sites.csv')
Spans = pd.read_csv('data/topology/Spans.csv')
Structures = pd.read_csv('data/topology/Structures.csv')
TransmissionLines = pd.read_csv('data/topology/Transmission_Lines.csv')

Offers201909 = pd.concat(pd.read_csv(f'data/generators/2019090{i}_Offers.csv') for i in range(1,8))
Bids201909 = pd.read_csv('data/historicaLMPs/201909_Final_prices.csv')

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

# import json
# with open('DictSimpNetwork.json', 'w') as fp:
#     json.dump(DictSimpNetwork, fp)

with open('/generators/generator_adjacency_matrix_dict.json') as f:
    generator_adjacency_matrix_dict = json.load(f)



SimpNodes = pd.read_csv('data/ABM/ABM_Nodes.csv')
SimpNetDetails = pd.read_csv('data/ABM/ABM_Network_details.csv')

plottype = '19 Nodes'
# plottype = 'All nodes'


if plottype == 'All nodes':

## Visualizing the location of all nodes and transmission lines
## Nodes
    plt.scatter(Sites.X.values, Sites.Y.values, marker = '.')
    # for i, txt in enumerate(Sites.MXLOCATION):
    #     if txt in SimpNetwork.SimpNode.values.tolist():
    #         plt.annotate(txt, (Sites.X[i], Sites.Y[i]), fontsize='large')

    ## Lines
    ## Preparing Transmission lines data

    TransmissionLines['MXLOC1'] = TransmissionLines['MXLOCATION'].swifter.apply(lambda x: x[:3])
    TransmissionLines['MXLOC2'] = TransmissionLines['MXLOCATION'].swifter.apply(lambda x: x[4:7])

    m = TransmissionLines.shape[0]

    Tlines = [None] * (2*(m))
    for l in range(m):
        MXLoc1 = TransmissionLines.MXLOC1.values[l]
        MXLoc2 = TransmissionLines.MXLOC2.values[l]
        if (MXLoc1 in Sites.MXLOCATION.values) & (MXLoc2 in Sites.MXLOCATION.values):
            loc1 = Sites[Sites.MXLOCATION == MXLoc1].index.values.astype(int)[0]
            loc2 = Sites[Sites.MXLOCATION == MXLoc2].index.values.astype(int)[0]
            Tlines[2*l] = (Sites.X[loc1], Sites.X[loc2])
            Tlines[2*l+1] = (Sites.Y[loc1], Sites.Y[loc2])

    Tlines = [i for i in Tlines if i]

    plt.plot(*Tlines)

    # Labels
    plt.xlabel('Longitude (X)')
    plt.ylabel('Latitude (Y)')
    plt.axis('equal')
    plt.show()

elif plottype == '19 Nodes':

    ## Visualizing the location of 19 nodes compared to the existing nodes

    # Nodes

    # Referenced nodes
    Node19 = np.zeros((20,2))
    locations = Sites.MXLOCATION.unique().tolist()
    locations.remove('WKM')
    locations.remove('HLY')

    for i, node in enumerate(DictSimpNetwork.keys()):
        if node in locations:
            Node19[i, 0] = Sites.X[Sites.MXLOCATION == node]
            Node19[i, 1] = Sites.Y[Sites.MXLOCATION == node]
        elif node == 'WKM':
            Node19[i, 0] = Sites.X[Sites.MXLOCATION == node] + 50000
            Node19[i, 1] = Sites.Y[Sites.MXLOCATION == node]
        elif node == 'HLY':
            Node19[i, 0] = Sites.X[Sites.MXLOCATION == node]
            Node19[i, 1] = Sites.Y[Sites.MXLOCATION == node] - 50000
        else:
            Node19[i, 0] = Sites.X[Sites.MXLOCATION.apply(lambda x: x in DictSimpNetwork[node])].mean()
            Node19[i, 1] = Sites.Y[Sites.MXLOCATION.apply(lambda x: x in DictSimpNetwork[node])].mean()

    # B_star
    srs = pd.Series(list(DictSimpNetwork.keys()))

    locHAY = srs[srs == 'HAY'].index.values.astype(int)[0]
    locTWZ = srs[srs == 'TWZ'].index.values.astype(int)[0]

    p = 0.25
    Node19[-1, 0] = (Node19[locHAY, 0] + p*(Node19[locTWZ, 0]-Node19[locHAY, 0]))
    Node19[-1, 1] = (Node19[locHAY, 1] + p*(Node19[locTWZ, 1]-Node19[locHAY, 1]))

    plt.scatter(Node19[:,0],Node19[:,1], marker = 'o')
    for i, node in enumerate(DictSimpNetwork.keys()):
        plt.annotate(node, (Node19[i, 0], Node19[i, 1]), fontsize='large')
    plt.annotate('B_star', (Node19[19, 0], Node19[19, 1]), fontsize='large')

    # plt.scatter(Sites.X.values, Sites.Y.values, marker = '.')

    ## Lines
    srs = pd.Series(list(DictSimpNetwork.keys()))

    # Referenced lines
    color ='b'

    m = SimpNetDetails.shape[0]
    Tlines = [None] * (3*m)
    for l in range(m - 2):
        MxLoc1 = SimpNetDetails.LEAVE.values[l]
        MxLoc2 = SimpNetDetails.ENTER.values[l]
        loc1 = srs[srs == MxLoc1].index.values.astype(int)[0]
        loc2 = srs[srs == MxLoc2].index.values.astype(int)[0]
        Tlines[3*l] = (Node19[loc1, 0], Node19[loc2, 0])
        Tlines[3*l + 1] = (Node19[loc1, 1], Node19[loc2, 1])
        Tlines[3*l + 2] = color

    # B_star lines
    locHAY = srs[srs == 'HAY'].index.values.astype(int)[0]
    locTWZ = srs[srs == 'TWZ'].index.values.astype(int)[0]

    Tlines[3 * (m-2)] = (Node19[locHAY, 0], Node19[-1, 0])
    Tlines[3 * (m-2) + 1] = (Node19[locHAY, 1], Node19[-1, 1])
    Tlines[3 * (m-2) + 2] = color
    Tlines[3 * (m-1)] = (Node19[locTWZ, 0], Node19[-1, 0])
    Tlines[3 * (m-1) + 1] = (Node19[locTWZ, 1], Node19[-1, 1])
    Tlines[3 * (m-1) + 2] = color

    plt.plot(*Tlines)
    add_arrow(*Tlines)
    Lines = *Tlines

    # Labels
    plt.xlabel('Longitude (X)')
    plt.ylabel('Latitude (Y)')
    plt.axis('equal')
    plt.show()

## Visualizing the location of 2 nodes

