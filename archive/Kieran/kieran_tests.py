import pandas as pd
import json
import numpy as np
import swifter

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


with open('data/generators/generator_adjacency_matrix_dict.json') as f:
    generator_adjacency_matrix_dict = json.load(f)



def get_key(val):
    for key, value in DictSimpNetwork.items():
        if val in value:
            return key

    return "key doesn't exist"


for gen in list(generator_adjacency_matrix_dict.keys()):
    generator_adjacency_matrix_dict[gen][0] = get_key(gen.split('_')[1][:3])

with open('generator_adjacency_matrix_dict1.json', 'w') as fp:
     json.dump(generator_adjacency_matrix_dict, fp)
