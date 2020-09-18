from lmpnz.GuillaumeExample import price_making_algorithm
from lmpnz.GuillaumeExample import economic_dispatch
from lmpnz.GuillaumeExample import price_taking_algorithm
import imp
imp.reload(economic_dispatch)
imp.reload(price_taking_algorithm)
import json
import math
import numpy as np
import pandas as pd
import pyomo.environ as pyo
import lmpnz.Network.PriceBids.Load.Load as ld
import stored_path
from lmpnz.GuillaumeExample import LMP
from lmpnz.Network.PriceBids.Generator.Generator import Generator
from lmpnz.Network.PriceBids.Load.Load import Load
from lmpnz.Network.Topology.Topology import Topology as top
import matplotlib.pyplot as plt


def tweak_d(d, load_factor = 1.3, index_to_tweak = 10, load_factor_for_node=12.1):
    save = d[index_to_tweak].copy()
    save_d = d.copy()
    d[index_to_tweak] = save
    d = load_factor * save_d.copy()
    d[index_to_tweak] = save * load_factor_for_node
    return d


def add_loads_to_topology(AMB_network):
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
    return AMB_network

def add_generators_to_topology(AMB_network):
    file_path = stored_path.main_path + '/data/generators/generator_adjacency_matrix_dict1.json'
    with open(file_path) as f:
        data = json.loads(f.read())

    number_of_added_generators = 0
    for name_generator in data.keys():
        L_ = data[name_generator]
        try:
            if type(L_[0]) != float:
                if not math.isnan(L_[-2]):
                    if L_[-1] == 'Hydro':
                        P_min = L_[-2]
                    else:
                        P_min = 0

                    g = Generator(name_generator, L_[0], 0, L_[-1], Pmax=L_[-2], Pmin=P_min,
                                  marginal_cost=np.array(L_[1]))
                    AMB_network.add_generator(g)
                    number_of_added_generators += 1
        except:
            pass

    return AMB_network


def get_load_matrix(AMB_network, day, Horizon_T):
    d = []
    for k, node in enumerate(AMB_network.loads.keys()):
        d.append([])
        for j in range(day * 48, day * 48 + Horizon_T):
            d[k].append(1 * 1000 * AMB_network.loads[node][0].return_d(1 + j // 48, j % 48 + 1))
    d = np.array(d)
    return d


def get_producers_matrices(AMB_network, day, Horizon_T):
    n_generator = AMB_network.get_number_of_gen()
    b = np.zeros((n_generator, Horizon_T))
    P_max = np.zeros((n_generator, Horizon_T))
    P_min = np.zeros((n_generator, Horizon_T))
    for node in AMB_network.generators.keys():
        for g in AMB_network.generators[node]:
            for i, j in enumerate(range(day * (48-1), day * (48-1) + Horizon_T)):
                if g.name == "diesel_gen":
                    pmax, pmin, a = 500, 0, 100
                else:
                    pmax, pmin, a = LMP.get_P_min_a(g.name, 1 + j // 48, j % 48 + 1, g.type)
                P_max[g.index, i] = pmax
                P_min[g.index, i] = pmin if g.type == "Hydro" else 0
                b[g.index, i] = a if a > 0 else np.random.randint(0, 50)
    return b, P_max, P_min


def get_basics(Horizon_T, day):
    AMB_network = top(network="ABM")
    AMB_network = add_loads_to_topology(AMB_network)
    AMB_network = add_generators_to_topology(AMB_network)
    H, h = AMB_network.H, AMB_network.h
    print("Topology loaded")
    """
    Tweak case  : add a fake generator
    """
    node_name = "MDN"
    AMB_network.add_generator(Generator("diesel_gen", node_name, 0, 0, Pmax=200, Pmin=0,
                                        marginal_cost=[0, 0]))

    """
    Get the load data
    """
    d = get_load_matrix(AMB_network, day, Horizon_T)
    d = tweak_d(d, load_factor = 1.3, index_to_tweak = 10, load_factor_for_node=12.1)
    # d = tweak_d(d, load_factor=1, index_to_tweak=10, load_factor_for_node=1)
    print("Load historical data loaded and tweaked")

    """
    Get the bid matrices
    """
    b, P_max, P_min = get_producers_matrices(AMB_network, day, Horizon_T)
    print("Load historical bids")

    """
    Load now the topology of generators
    """
    Mn = AMB_network.Mn
    return H, h, Mn, b, P_max, P_min, d


def baseline_prices():
    Horizon_T, day = 48, 2
    H, h, Mn, b, P_max, P_min, d = get_basics(Horizon_T, day)
    n = d.shape[0]  # number of nodes

    """
    Find lambdas for the day (they will be deemed exogenous)
    """
    lambdas = np.zeros((n, Horizon_T))
    gammas = np.zeros(Horizon_T)
    for j in range(Horizon_T):
        """
        Here is a new optimization framework which is rigoursely the same as devised in the algorithm, 
        WARNING this is just for time period j.

        We input the c and q, the price and quantity offered by the battery. Here 0,0 because we want the LMPs
        without the battery
        """
        c = 0
        q = 0
        model = economic_dispatch.run_ED_1period(d[:, j], b[:, j], P_max[:, j], P_min[:, j], c, q, H, h, Mn)
        for k in range(n):
            lambdas[k, j] = model.dual[model.injection_definition[k]]  # here we store the dual variables of the injection definition constraint

        gammas[j] = model.dual[model.injection_balance]

    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, sharex=True, figsize=(15, 10), dpi=80, facecolor='w', edgecolor='k')
    plt.subplots_adjust(hspace=.0)
    fs = 20

    # if Node is not None:
    #     Node_name = Nodes[Node]
    #     Nodes = [None] * len(Nodes)
    #     Nodes[Node] = Node_name
    #     Y = np.array(LMP_df[[f'{i}' for i in range(t)]][1:]).T - np.array([gamma_df.gamma[:t].tolist()]).T
    # else:
    #     Y = np.array(LMP_df[[f'{i}' for i in range(t)]][1:]).T - np.array([gamma_df.gamma[:t].tolist()]).T

    axs[0].plot(gammas)
    axs[0].set_ylabel('Average price $\gamma$ [\$/MW]', fontsize=fs)
    axs[0].set_ylim(bottom=0)
    axs[0].set_title('Average price and congestion curves\n Baseline model, Sept. 2nd, 2019', fontsize=fs)
    axs[0].grid()

    for i in range(1, 20):
        axs[1].plot(lambdas[i]-gammas, label=f'{i}')
    # for i, y_arr, label in zip(range(1, 20), Y[1:, :], Nodes[1:].tolist()):
    #     if (label == 'MDN') | (label == 'HEN'):
    #         axs[1].plot(y_arr, label=f'{i} : {label}', linewidth=5)
    #     else:
    #         axs[1].plot(y_arr, label=f'{i} : {label}')

    axs[1].legend()
    axs[1].set_ylabel('$\lambda - \gamma$ [\$/MW]', fontsize=fs)
    axs[1].set_xlabel('Time [h]', fontsize=fs)
    plt.xticks(range(0, 48, 2), range(24))
    axs[1].set_xlim([0, 48])
    axs[1].grid()
    plt.show()