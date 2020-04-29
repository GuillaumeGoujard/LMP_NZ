import pandas as pd
import numpy as np
import swifter

SimpNetwork = pd.read_csv('data/ABM/ABM_Simplified_network.csv')
SimpNetwork.rename(columns = {
    'Swem Node' : 'SimpNode',
    ' NZEM Substations that act as Grid Exit Points' : 'OriginNodes'
}, inplace=True)

Offers201909 = pd.concat(pd.read_csv(f'data/generators/2019090{i}_Offers.csv') for i in range(1,8))

# dropping reserve offers
Offers201909 = Offers201909[Offers201909['ProductType'] != 'Reserve']

# drop units that bid 0 MW for all 5 bands in the last timestamp of the tp
Offers201909 = Offers201909[Offers201909['Megawatt'] != 0]

DictSimpNetwork = {
    snode: list(set([onode[:3] for onode in SimpNetwork.OriginNodes[SimpNetwork.SimpNode == snode].values[0].split(' ')[1:]])) for snode in SimpNetwork.SimpNode
}

def getKeysByValue(dictOfElements, valueToFind):
    listOfKeys = list()
    listOfItems = dictOfElements.items()
    for item  in listOfItems:
        if item[1] == valueToFind:
            listOfKeys.append(item[0])
    return  listOfKeys

gen_names = Offers201909.Unit.unique().tolist()

orignodes = pd.Series(Offers201909.PointOfConnection[Offers201909.Unit == unit].unique().tolist()
     for unit in gen_names).swifter.apply(lambda x: x[0][:3])

simpnodes = pd.Series(Offers201909.PointOfConnection[Offers201909.Unit == unit].unique().tolist()
     for unit in gen_names).swifter.apply(lambda x: x[0][:3]).swifter.apply(lambda x: [key for (key, value) in DictSimpNetwork.items() if x in value]).tolist()
simpnodes = pd.Series(simpnodes).apply(lambda x: x[0] if len(x) == 1 else '')

# df = pd.DataFrame({
#     'gen_names' : gen_names,
#     'orignodes' : orignodes,
#     'simpnodes' : simpnodes
# })

A = [None] * len(gen_names)
for i in range(len(A)):
    gen = gen_names[i]

    A[i] = round()

Pmins =

Pmaxs =

fuel_names =

Gendict = {
    gen_name : [node[0], [a,0], Pmin, Pmax, fuel_name]
              for gen_name, node, a, Pmin, Pmax, fuel_name
              in zip(gen_names, simpnodes, A, Pmins, Pmaxs, fuel_names)
}