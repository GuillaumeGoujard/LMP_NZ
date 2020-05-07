from typing import Tuple

import numpy as np
import pandas as pd

import main.Network.PriceBids.Load.Load as ld
import stored_path
from main.Network.PriceBids.Generator.Generator import Generator
from main.Network.PriceBids.Load.Load import Load


class Topology:
    """
    For the three dictionnaries : nodes, generators, loads they are linking the index (int) of the node to the name (str)
    of node, generator or load

    f_nodes_2_names takes the index of the dict and outputs a list of name of nodes
    f_names_to_nodes
    """
    def __init__(self, f_names_2_nodes=None, network=''):
        if f_names_2_nodes is None:
            f_names_2_nodes = dict()

        if network is "ONDE":
            f_names_2_nodes = {"NDE": 0}
            self.names_2_nodes = f_names_2_nodes
            self.nodes_2_names = generate_nodes_2_names(f_names_2_nodes)
            self.A = None
            self.I = None
            self.H = None
            self.h = None

        elif network is "NSNDE":
            Nodes = np.array(['NTH','STH'])

            # Creating the network
            Network = pd.DataFrame()
            Network['LEAVE'] = ['NTH','STH']
            Network['ENTER'] = ['STH','NTH']
            m = Network.shape[0]
            Network['NLeave'] = np.array([np.where(Nodes == Network['LEAVE'][l])[0][0] for l in range(m)])
            Network['NEnter'] = np.array([np.where(Nodes == Network['ENTER'][l])[0][0] for l in range(m)])

            Network['Resistance (Ohms)'] = [0.000098,0.000098]
            Network["Reactance (Ohms)"] = [0.01,0.01]
            Network["Capacity(MW)"] = [700,700]

            # Preliminary characteristics
            f_names_2_nodes = dict([[node, j] for j, node in enumerate(Nodes)])
            self.names_2_nodes = f_names_2_nodes
            self.nodes_2_names = generate_nodes_2_names(self.names_2_nodes)
            self.I = create_incidence(Network.NLeave, Network.NEnter)
            self.A = create_adjacency(Network.NLeave, Network.NEnter)

            # Line characteristics
            omega_NZ = 50 * (2 * np.pi)
            z = Network['Resistance (Ohms)'] + 1j * Network["Reactance (Ohms)"] * omega_NZ
            y = 1 / z
            self.y = np.imag(y)
            self.H = create_H(self.I, self.y)
            self.h = pd.concat([Network["Capacity(MW)"], Network["Capacity(MW)"]]).values

        elif network is "ABM":
            Network = pd.read_csv(stored_path.main_path + '/data/ABM/ABM_Network_details.csv')
            Nodes = np.unique(np.concatenate((np.unique(Network.LEAVE), np.unique(Network.ENTER))))
            Nodes[0], Nodes[1] = Nodes[1], Nodes[0]
            m = Network.shape[0]
            Network['NLeave'] = np.array([np.where(Nodes == Network['LEAVE'][l])[0][0] for l in range(m)])
            Network['NEnter'] = np.array([np.where(Nodes == Network['ENTER'][l])[0][0] for l in range(m)])

            self.names_2_nodes = dict([[node, j] for j, node in enumerate(Nodes)])
            self.nodes_2_names = generate_nodes_2_names(self.names_2_nodes)
            self.I = create_incidence(Network.NLeave, Network.NEnter)
            self.A = create_adjacency(Network.NLeave, Network.NEnter)
            self.test = 0

            omega_NZ = 50*(2*np.pi)
            z = Network['Resistance (Ohms)'] + 1j*Network["Reactance (Ohms)"]*omega_NZ
            y = 1/z
            self.y = np.imag(y)
            self.H = create_H(self.I, self.y)
            self.h = pd.concat([Network["Capacity(MW)"], Network["Capacity(MW)"]]).values

        else:
            self.names_2_nodes = f_names_2_nodes
            self.nodes_2_names = generate_nodes_2_names(f_names_2_nodes) if f_names_2_nodes is not None else 0
            self.A = None
            self.I = None
            self.H = None
            self.h = None

        self.number_nodes = self.nodes_2_names.keys().__len__()

        self.loads = dict([[node, []] for node in self.nodes_2_names.keys()])
        self.load_data = 0

        self.number_generators = 0
        self.generators = dict([[node, []] for node in self.nodes_2_names.keys()])

        self.Pmin = None
        self.Pmax = None
        self.Mn = None
        self.Qt = None
        self.at = None
        self.Ag = None
        self.qg = None
        self.Au = None
        self.xt = None
        self.ct = None

    ## Methods

    # Adding elements
    def add_nodes(self, node_name):

        return

    def add_lines(self, Leave_node_name, Enter_node_name):

        return

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
        return self.loads

    def add_battery(self):
        return

    # Creating elements
    def create_Mn(self) -> np.array:
        Mn = np.zeros((self.number_nodes, self.number_generators))
        for k in self.generators.keys():
            for l in self.generators[k]:
                Mn[k][l.index] = 1
        self.Mn = Mn
        return self.Mn

    def get_number_of_gen(self):
        g = 0
        for node in self.generators.keys():
            g += len(self.generators[node])
        return g

    def create_H_h(self) -> np.array:
        self.H = create_H(self.I, self.y)
        self.h = pd.concat([self.something, self.something]).values
        return self.H, self.h

    def create_Pmin_Pmax(self):
        self.Pmin = np.array([[g.Pmin for g in list(get_all_values(self.generators))]]).T
        self.Pmax = np.array([[g.Pmax for g in list(get_all_values(self.generators))]]).T
        return self.Pmin, self.Pmax

    def create_Ag_qg(self) -> Tuple[np.array, np.array]:
        self.Ag = np.concatenate((np.eye(self.number_generators),-np.eye(self.number_generators))
                            ,axis = 0)
        self.qg = np.concatenate((self.Pmax, -self.Pmin), axis = 0)
        return self.Ag, self.qg

    def create_Qt_at(self):
        self.Qt = np.diag([g.q for g in list(get_all_values(self.generators))])
        self.at = np.array([[g.a for g in list(get_all_values(self.generators))]]).T
        return self.Qt, self.at

    def create_Au_xt_ct(self):
        self.Au = np.zeros((2*self.number_nodes,self.number_nodes))
        self.xt = np.zeros((self.number_nodes,1))
        self.ct = np.zeros((self.number_nodes,1))
        return self.Au, self.xt, self.ct


        # Other functions

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

def create_H(I, y):
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

    # Network = pd.read_csv(stored_path.main_path + '/data/ABM/ABM_Network_details.csv')
    # Nodes = np.unique(np.concatenate((np.unique(Network.LEAVE), np.unique(Network.ENTER))))
    # Nodes[0], Nodes[1] = Nodes[1], Nodes[0]
    # m = Network.shape[0]
    # Network['NLeave'] = np.array([np.where(Nodes == Network['LEAVE'][l])[0][0] for l in range(m)])
    # Network['NEnter'] = np.array([np.where(Nodes == Network['ENTER'][l])[0][0] for l in range(m)])
    #
    # names_2_nodes = dict([[node, j] for j, node in enumerate(Nodes)])
    # nodes_2_names = generate_nodes_2_names(names_2_nodes)
    # I = create_incidence(Network.NLeave, Network.NEnter)
    # A = create_adjacency(Network.NLeave, Network.NEnter)
    #
    # omega_NZ = 50 * (2 * np.pi)
    # z = Network['Resistance (Ohms)'] + 1j * Network["Reactance (Ohms)"] * omega_NZ
    # y = 1 / z
    # y = np.imag(y)

    m = I.shape[1]
    Delta_y = np.diag(y)

    Y = I @ Delta_y @ I.T

    Y_bar = Y.copy()
    Y_bar[0,:] = 0
    Y_bar[:,0] = 0

    Y_bar = np.delete(np.delete(Y, 0, 0), 0, 1)

    Y_dag = np.concatenate((np.zeros((1, Y_bar.shape[0] + 1)),
                            np.concatenate((np.zeros((Y_bar.shape[0], 1)),
                                            np.linalg.inv(Y_bar)),
                                           axis=1)),
                           axis=0)

    H_hat = np.diag(y) @ I.T @ Y_dag

    H = np.concatenate((np.eye(m), -np.eye(m)), axis=0) @ H_hat

    return H

def create_H_hat(I, y):
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

    Y_bar = Y.copy()
    Y_bar[0,:] = 0
    Y_bar[:,0] = 0

    Y_bar = np.delete(np.delete(Y, 0, 0), 0, 1)

    Y_dag = np.concatenate((np.zeros((1, Y_bar.shape[0] + 1)),
                            np.concatenate((np.zeros((Y_bar.shape[0], 1)),
                                            np.linalg.inv(Y_bar)),
                                           axis=1)),
                           axis=0)

    H_hat = np.diag(y) @ I.T @ Y_dag

    return H_hat

def Aq_matrix():
    return


def get_all_values(d):
    if isinstance(d, dict):
        for v in d.values():
            yield from get_all_values(v)
    elif isinstance(d, list):
        for v in d:
            yield from get_all_values(v)
    else:
        yield d


################## Test code ##################

if __name__ == '__main__':
    """
    Test : Custom network creation
    """
    test_node = {
                     "AA1":0,
                     "AA2":0,
                     "AA3":0,
                     "BB1":1,
                     "BB2":1,
                 }
    top = Topology(f_names_2_nodes = test_node)

    """
    Test : One Node network
    """
    OneNode_network = Topology(network="One node")

    """
    Test : Two Node network
    """
    TwoNode_network = Topology(network='North-South node')

    """
    Test : AMB Network
    """
    AMB_network = Topology(network="ABM")

    # AMB_network.names_2_nodes

    """
    Create two generators and add them to the network
    """
    # g = Generator("GuillaumeGenerator", "HEN", 0, "dummy", Pmax=20, Pmin=0, marginal_cost=[10,0])
    # AMB_network.add_generator(g)
    # a = Generator("AliceGenerator", "HEN", 2, "dummy", Pmax=200, Pmin=0, marginal_cost=[200,0])
    # AMB_network.add_generator(a)
    # k = Generator("KieranGenerator", "MAN", 1, "dummy", Pmax=100, Pmin=5, marginal_cost=[50,0])
    # AMB_network.add_generator(k)

    """
    Create loads on each node
    """
    Existing_sub_nodes = ld.get_existing_subnodes()
    historical_loads = ld.get_historical_loads()
    Simp_nodes_dict = ld.get_nodes_to_subnodes()
    Simp_nodes_dict["MAN"] = ["MAN2201"]
    Existing_sub_nodes.append("MAN2201")
    nodes_to_index = pd.read_csv(stored_path.main_path + '/data/ABM/ABM_Nodes.csv')
    for i, node in enumerate(AMB_network.names_2_nodes.keys()):
        # print("Load added at node : " + node)
        index = nodes_to_index[nodes_to_index["Node names"] == node]["Node index"].values[0]
        load = Load(node, node, index, type="real_load")
        load.add_load_data(historical_loads, Simp_nodes_dict, Existing_sub_nodes)
        AMB_network.add_load(load)

    """
    get d_t for day 12 and trading period 1
    """
    d = []
    for node in AMB_network.loads.keys():
        d.append(AMB_network.loads[node][0].return_d(12,1))


    """
    Add generators
    """
    import json
    import math
    file_path = stored_path.main_path + '/data/generators/generator_adjacency_matrix_dict.json'
    with open(file_path) as f:
        data = json.loads(f.read())

    number_of_added_generators = 0
    for name_generator in data.keys():
        L = data[name_generator]
        try:
            if type(L[0]) != float:
                if not math.isnan(L[-2]):
                    g = Generator(name_generator, L[0], 0, L[-1], Pmax=L[-2], Pmin=L[-3], marginal_cost=L[1])
                    AMB_network.add_generator(g)
                    number_of_added_generators +=1
        except:
            pass

    """
    Add topology specific characteristics
    """
    AMB_network.create_Pmin_Pmax()
    AMB_network.create_Qt_at()







