import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as lines
import numpy as np
import swifter

# Importing relevant csv files
Sites = pd.read_csv('data/topology/Sites.csv')
Spans = pd.read_csv('data/topology/Spans.csv')
Structures = pd.read_csv('data/topology/Structures.csv')
TransmissionLines = pd.read_csv('data/topology/Transmission_Lines.csv')

Offers201909week1 = pd.concat(pd.read_csv(f'data/generators/2019090{i}_Offers.csv') for i in range(1,8))
Bids201909 = pd.read_csv('data/historicaLMPs/201909_Final_prices.csv')

SimpNetwork = pd.read_csv('data/ABM/ABM_Simplified_network.csv')
SimpNetwork.rename(columns = {
    'Swem Node' : 'SimpNode',
    ' NZEM Substations that act as Grid Exit Points' : 'OriginNodes'
}, inplace=True)

SimpNodes = pd.read_csv('data/ABM/ABM_Nodes.csv')
SimpNetDetails = pd.read_csv('data/ABM/ABM_Network_details.csv')

# Visualizing the location of all nodes
# Nodes
plt.scatter(Sites.X.values, Sites.Y.values, marker = '.')

# Lines
# Preparing Transmission lines data

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

## Visualizing the location of 19 nodes
for i, txt in enumerate(Sites.MXLOCATION):
    if txt in SimpNetwork['Swem Node'].values.tolist():
        plt.annotate(txt, (Sites.X[i], Sites.Y[i]))

DictSimpNetwork = {
    snode: list(set([onode[:3] for onode in SimpNetwork.OriginNodes[SimpNetwork.SimpNode == snode].values[0].split(' ')[1:]])) for snode in SimpNetwork.SimpNode
}

# Nodes
m = 19

Node19 = np.zeros((19,2))
for i, node in enumerate(DictSimpNetwork.keys()):
    Node19[i, 0] = Sites.X[Sites.MXLOCATION.apply(lambda x: x in DictSimpNetwork[node])].mean()
    Node19[i, 1] = Sites.Y[Sites.MXLOCATION.apply(lambda x: x in DictSimpNetwork[node])].mean()

plt.scatter(Node19[:,0],Node19[:,1], marker = '.')

Tlines = [None] * (2*(19))
for l in range(m):
    MXLoc1 = TransmissionLines.MXLOC1.values[l]
    MXLoc2 = TransmissionLines.MXLOC2.values[l]
    loc1 = Sites[Sites.MXLOCATION == MXLoc1].index.values.astype(int)[0]
    loc2 = Sites[Sites.MXLOCATION == MXLoc2].index.values.astype(int)[0]
    Tlines[2*l] = (Sites.X[loc1], Sites.X[loc2])
    Tlines[2*l+1] = (Sites.Y[loc1], Sites.Y[loc2])

Tlines = [i for i in Tlines if i]

plt.plot(*Tlines)

for i, node in enumerate(DictSimpNetwork.keys()):
    plt.annotate(node, (Node19[i, 0], Node19[i, 1]))

# Labels
plt.xlabel('Longitude (X)')
plt.ylabel('Latitude (Y)')
plt.axis('equal')
plt.show()

## Visualizing the location of 2 nodes


## Visualizing location of 1 node