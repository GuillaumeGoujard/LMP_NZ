from datetime import datetime as datetime
from typing import List, Tuple, Dict
import numpy as np
import pandas as pd
from main.Network.PriceBids.Generator.Generator import Generator
from main.Network.PriceBids.Load.Load import Load
import os
dir_path = os.path.dirname(os.path.realpath(__file__))
root = os.path.dirname(os.path.dirname(os.path.dirname(dir_path)))



class Topology:
    """
    For the three dictionnaries : nodes, generators, loads they are linking the index (int) of the node to the name (str)
    of node, generator or load

    f_nodes_2_names takes the index of the dict and outputs a list of name of nodes
    f_names_to_nodes
    """
    def __init__(self, f_names_2_nodes=None, network=None):
        if network is "ABM":
            path = root + "/data/ABM/ABM_Network_details.csv"
            Network = pd.read_csv(path)
            Nodes = np.unique(np.concatenate((np.unique(Network.LEAVE), np.unique(Network.ENTER))))
            m = Network.shape[0]
            Network['NLeave'] = np.array([np.where(Nodes == Network['LEAVE'][l])[0][0] for l in range(m)])
            Network['NEnter'] = np.array([np.where(Nodes == Network['ENTER'][l])[0][0] for l in range(m)])
            self.names_2_nodes = dict([[node, j] for j, node in enumerate(Nodes)])
            self.nodes_2_names = generate_nodes_2_names(self.names_2_nodes )
            self.I = create_incidence(Network.NLeave, Network.NEnter)
            self.A = create_adjacency(Network.NLeave, Network.NEnter)
            omega_NZ = 50*(2*np.pi)
            z = Network['Resistance (Ohms)'] + 1j*Network["Reactance (Ohms)"]*omega_NZ
            y = 1/z
            y = y.imag
            self.H = H_matrix(self.I, y)
            self.h = pd.concat([Network["Capacity(MW)"], Network["Capacity(MW)"]]).values
        else:
            self.names_2_nodes = f_names_2_nodes
            self.nodes_2_names = generate_nodes_2_names(f_names_2_nodes)
            self.A = None
            self.H = None
            self.I = None
            self.h = None

        self.number_nodes = self.A.shape[0] if self.A is not None else 0
        self.Mn = None
        self.generators = dict([[node, []] for node in self.nodes_2_names.keys()])
        self.loads = dict([[node, []] for node in self.nodes_2_names.keys()])
        self.number_generators = 0


    def add_generator(self, generator: Generator):
        node = self.names_2_nodes[generator.node_name]
        generator.index = self.number_generators
        self.generators[node].append(generator)
        self.number_generators += 1
        self.create_Mn()
        return self.generators

    def add_load(self, load: Load):
        node = self.names_2_nodes[load.node_name]
        self.loads[node].append(load)
        return self.generators

    def create_Mn(self) -> np.array:
        Mn = np.zeros((self.number_nodes, self.number_generators))
        for k in self.generators.keys():
            for l in self.generators[k]:
                Mn[k][l.index] = 1
        self.Mn = Mn
        return self.Mn

    def create_H_h(self, input_line_data: pd.DataFrame) -> np.array:

        return self.H, self.h

    def create_Ag_qg(self) -> Tuple[np.array, np.array]:
        Ag, qg = None, None
        return Ag, qg


def generate_nodes_2_names(f_names_2_nodes):
    output = dict([[k, []] for k in set(f_names_2_nodes.values())])
    for v in f_names_2_nodes.keys():
        output[f_names_2_nodes[v]].append(v)
    return output


def create_adjacency(NLeave, NEnter):
    m = NLeave.shape[0]
    n = np.unique(np.concatenate((NLeave, NEnter))).size
    A = np.zeros((n, n))

    for l in range(m):
        A[NLeave[l], NEnter[l]] = 1
        A[NEnter[l], NLeave[l]] = 1

    return A


def create_incidence(NLeave, NEnter):
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

    return M


def H_matrix(I, y):
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
    m = I.shape[1]
    Delta_y = np.diag(y)

    Y = I @ Delta_y @ I.T

    Y_bar = np.delete(np.delete(Y, 0, 0), 0, 1)

    Y_dag = np.concatenate((np.zeros((1, Y_bar.shape[0] + 1)),
                            np.concatenate((np.zeros((Y_bar.shape[0], 1)),
                                            np.linalg.inv(Y_bar)),
                                           axis=1)),
                           axis=0)

    H_hat = np.diag(y) @ I.T @ Y_dag

    H = np.concatenate((np.eye(m), -np.eye(m)), axis=0) @ H_hat

    return H



if __name__ == '__main__':
    """
    Test 1
    """
    test_node = {
                     "AA1":0,
                     "AA2":0,
                     "AA3":0,
                     "BB1":1,
                     "BB2":1,
                 }
    top = Topology(test_node)

    """
    Test 2
    """
    AMB_network = Topology(network="ABM")

    # AMB_network.names_2_nodes

    """
    Create two generators and add them to the network
    """
    g = Generator("GuillaumeGenerator", "HEN", 0, "dummy", Pmax=20, Pmin=0, marginal_cost=10)
    AMB_network.add_generator(g)
    a = Generator("AliceGenerator", "HEN", 2, "dummy", Pmax=200, Pmin=0, marginal_cost=200)
    AMB_network.add_generator(a)
    k = Generator("KieranGenerator", "MAN", 1, "dummy", Pmax=100, Pmin=5, marginal_cost=50)
    AMB_network.add_generator(k)

    """
    Create loads on each nodes 
    """
    for i, node in enumerate(AMB_network.names_2_nodes.keys()):
        print("Load added at node : " + node)
        AMB_network.add_load(Load(name=str(i), node_name=node, index=i, type="dummy", constant_demand=1))