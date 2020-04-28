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
SimpNodes = pd.read_csv('data/ABM/ABM_Nodes.csv')
SimpNetDetails = pd.read_csv('data/ABM/ABM_Network_details.csv')

## Visualizing the location of all nodes

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
    node: [] for node in
}

## Visualizing the location of 2 nodes


## Visualizing location of 1 node