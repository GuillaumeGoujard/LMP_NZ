from datetime import datetime as datetime
from typing import List, Tuple, Dict
import numpy as np
import pandas as pd
from main.Network.PriceBids.Generator.Generator import Generator
from main.Network.PriceBids.Load.Load import Load

def generate_nodes_2_names(f_names_2_nodes):
    output = dict([[k,[]] for k in set(f_names_2_nodes.values())])
    for v in f_names_2_nodes.keys():
        output[f_names_2_nodes[v]].append(v)
    return output


class Topology:
    """
    For the three dictionnaries : nodes, generators, loads they are linking the index (int) of the node to the name (str)
    of node, generator or load

    f_nodes_2_names takes the index of the dict and outputs a list of name of nodes
    f_names_to_nodes
    """
    def __init__(self, f_names_2_nodes):
        self.names_2_nodes = f_names_2_nodes
        self.nodes_2_names = generate_nodes_2_names(f_names_2_nodes)
        self.generators = dict([[node, []] for node in self.nodes_2_names.keys()])
        self.loads = dict([[node, []] for node in self.nodes_2_names.keys()])
        self.Mn = None
        self.H = None
        self.h = None

    def add_generator(self, generator: Generator):
        node = self.names_2_nodes[generator.node_name]
        self.generators[node].append(generator)
        return self.generators

    def add_load(self, load: Load):
        node = self.names_2_nodes[load.node_name]
        self.loads[node].append(load)
        return self.generators

    def create_Mn(self) -> np.array:

        return self.Mn

    def create_H_h(self, input_line_data: pd.DataFrame) -> np.array:

        return self.H, self.h

    def H_matrix(self, NLeave, NEnter, y):
        '''
        Shift Factor matrix

        Args:
            NLeave: Leaving node vector
                Size: (m,)
                Type: np.array
                Unit: integer value
                Description: List of nodes through which line of index l leaves

            NEnter: Entering node vector
                Size: (m,)
                Type: np.array
                Unit: integer value
                Description: List of nodes through which line of index l enters

            y: Impedance vector
                Size: (m,)
                Type: np.array
                Unit: Ohms
                Description: Array of impedances of each line, in Ohms. Each row corresponds to the indice of line l.

        Returns:
            H: Shift factor matrix
                Size: (2*m,n)
                Type: np.array
                Unit: ???
                Description:

        '''
        # Initializing sizes
        m = NLeave.size
        n = np.unique(np.concatenate((NLeave, NEnter))).size

        N = np.unique(np.concatenate((NLeave, NEnter)))

        # Initializing M
        M = np.zeros((n, m))

        # Creating M
        for i in range(n):
            for l in range(m):
                if NLeave[l] == N[i]:
                    M[i, l] = 1
                elif NEnter[l] == N[i]:
                    M[i, l] = -1

        Delta_y = np.diag(y)

        Y = M @ Delta_y @ M.T

        Y_bar = np.delete(np.delete(Y, 0, 0), 0, 1)

        Y_dag = np.concatenate((np.zeros((1, Y_bar.shape[0] + 1)),
                                np.concatenate((np.zeros((Y_bar.shape[0], 1)),
                                                np.linalg.inv(Y_bar)),
                                               axis=1)),
                               axis=0)

        H_hat = np.diag(y) @ M.T @ Y_dag

        H = np.concatenate((np.eye(m), -np.eye(m)), axis=0) @ H_hat

        return H

    def create_Ag_qg(self) -> Tuple[np.array, np.array]:
        Ag, qg = None, None
        return Ag, qg


if __name__ == '__main__':
    test_node = {
                     "AA1":0,
                     "AA2":0,
                     "AA3":0,
                     "BB1":1,
                     "BB2":1,
                 }
    top = Topology(test_node)