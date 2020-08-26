import article_example.dataPackage as dataPackage
from main.GuillaumeExample import price_making_algorithm
from main.GuillaumeExample import economic_dispatch
from main.GuillaumeExample import price_taking_algorithm
from article_example.averagecongestion import get_average_congestion_charge
import imp
import article_example.NZGridVizualization as nzgridvisu
imp.reload(economic_dispatch)
imp.reload(price_taking_algorithm)
import json
import math
import numpy as np
import pandas as pd
import pyomo.environ as pyo
import main.Network.PriceBids.Load.Load as ld
import stored_path
from main.GuillaumeExample import LMP
from main.Network.PriceBids.Generator.Generator import Generator
from main.Network.PriceBids.Load.Load import Load
import matplotlib.pyplot as plt

from main.Network.Topology.Topology import Topology as topology
import main.Network.Topology.Topology as Topology

import article_example.plotPrograms as plotprog


def baseline_prices(day, Horizon_T,  H, h, Mn, b, P_max, P_min, d):
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

    return lambdas, gammas


if __name__ == '__main__':
    day = 2
    Horizon_T = 48

    """
    First:
        - create a topology object "AMB network" which is loading the NewZealand grid topology
        - From the object, get the Shift-factor matrix H and line capacity limits h
    """
    AMB_network = topology(network="ABM")
    H, h = AMB_network.H, AMB_network.h

    """
    Then:
        - aggregate and add the 19 loads to "AMB network" 
        - aggregate and add the generators to "AMB network" 
        
    """
    AMB_network = Topology.add_loads_to_topology(AMB_network)
    AMB_network = Topology.add_generators_to_topology(AMB_network)

    #plot the grid
    nzgridvisu.plot_19_nodes()
    print("Topology loaded")


    """
    Get the load data
    """
    d = dataPackage.get_load_matrix(AMB_network, day, Horizon_T)
    plotprog.plot_nodal_demand(d)
    print("load historical data loaded")

    """
    Get the bid matrices from historical bid 
    """
    b, P_max, P_min = dataPackage.get_producers_matrices(AMB_network, day, Horizon_T, random_a=False)
    print("historical bids loaded")

    """
    Load now the topology of generators
    """
    Mn = AMB_network.Mn

    """
    Find the load factor for node HEN/12 (Auckland) to produce a congestion such that the congestion charge 
    is in average equal to the actual average congestion charge 
    """
    average_congestion = get_average_congestion_charge(node="HEN0331")

    """
    Get LMP without Battery and plot them
    """
    def f(lf, node_to_tweak=12, average_congestion=average_congestion):
        d = dataPackage.get_load_matrix(AMB_network, day, Horizon_T)
        d = dataPackage.tweak_d(d, load_factor=1, index_to_tweak=node_to_tweak, load_factor_for_node=lf)
        lambdas, gammas = baseline_prices(day, Horizon_T, H, h, Mn, b, P_max, P_min, d)
        return np.mean(abs(lambdas[node_to_tweak] - gammas)) - average_congestion

    from scipy import optimize
    root = optimize.bisect(lambda x: f(x, node_to_tweak=12, average_congestion=average_congestion), 1, 2.6, xtol=0.05)

    d = dataPackage.get_load_matrix(AMB_network, day, Horizon_T)
    d = dataPackage.tweak_d(d, load_factor=1, index_to_tweak=12, load_factor_for_node=root)

    """
    Adjust the bids so that the simulated LMPs are the actual LMPs
    """
    lambdas, gammas = baseline_prices(day, Horizon_T, H, h, Mn, b, P_max, P_min, d)

    actual_lmps = pd.read_csv("Wholesale_price_trends_20200825111248.csv")
    actual_system_lambda = actual_lmps["Price ($/MWh)"].values
    offset = actual_system_lambda - gammas


    lambdas, gammas = baseline_prices(day, Horizon_T, H, h, Mn, b+offset, P_max, P_min, d)
    plotprog.plot_lambdas_gammas(lambdas, gammas)
    b = b+offset

    # congestions = {}
    # for t in range(d.shape[1]):
    #     congestions[t] = []
    #     for i in range(d.shape[0]):
    #         if (abs(lambdas[i,t]-gammas[t]) > 5):
    #             congestions[t].append((i, abs(lambdas[i,t]-gammas[t])))
    #             # print("time {} node {}".format(t, i))



    """
    Influence investment over capacity
    """
    y_s = list(range(1,10))  # amortize in 10 years
    sequence_of_costs = [200 * 1000 / (y * 365) for y in y_s]

    # caps = []
    # for c in sequence_of_costs:
    #     model = price_making_algorithm.run_program(d, b, P_max, P_min, H, h, Mn, i_battery=12,
    #                                                max_capacity=None, cost_of_battery=c)
    #     caps.append(pyo.value(model.z_cap))

    caps_south = []
    for c in sequence_of_costs:
        model = price_making_algorithm.run_program(d, b, P_max, P_min, H, h, Mn, i_battery=1,
                                                   max_capacity=None, cost_of_battery=c)
        caps_south.append(pyo.value(model.z_cap))

    #empirically select
    y = 6



    #
    # model.z_cap.pprint()  # To print the content of a variable or constraint do model.[name of variable].pprint()
    # # # to see all the variables open the workspace and see all of the variables model contains
    # # z_cap_store = pyo.value(model.z_cap)  # use pyo.value to store
    # # q_u_store = [pyo.value(model.q_u[t]) for t in range(Horizon_T)]  # or do that for array variable
    # lambda_ = [[pyo.value(model.lambda_[i, t]) for t in range(Horizon_T)] for i in range(d.shape[0])]
    # plt.plot(np.mean(lambda_,axis=0))
    # plt.show()
    #
    # benef = -pyo.value(model.obj)  # model.obj = -lambda*u + B*z
    # arbitrage = -pyo.value(model.obj) + 55 * z_cap_store  # -model.obj + Bz = lambda*u - B*z + B*z
    # gamma_ = np.array([pyo.value(model.gamma_[t]) for t in range(Horizon_T)])
    #
    # plt.plot(lambda_[1] - gamma_)
    # plt.show()
    # """
    # You can also save the results using save_results function (as we did last week)
    # """
    y = 6  # amortize in 10 years
    cost_of_battery = 200 * 1000 / (y * 365)

    print("Launching model...")
    model = price_making_algorithm.run_program(d, b, P_max, P_min, H, h, Mn, i_battery=10,
                                               max_capacity=None, cost_of_battery=cost_of_battery)
    model.z_cap.pprint()
    print("Model computed")
    # save_results(model, d, Horizon_T, i_test=0)  # i-test is the number of the folder in which you store the results

    data = []
    for i in range(1, d.shape[0]):
        print(i)
        model = price_making_algorithm.run_program(d, b, P_max, P_min, H, h, Mn, i_battery=i,
                                                   max_capacity=None, cost_of_battery=cost_of_battery)
        z_cap_store = pyo.value(model.z_cap)  # use pyo.value to store
        q_u_store = [pyo.value(model.q_u[t]) for t in range(Horizon_T)]  # or do that for array variable
        lambda_ = [[pyo.value(model.lambda_[i, t]) for t in range(Horizon_T)] for i in range(d.shape[0])]

        benef = -pyo.value(model.obj)  # model.obj = -lambda*u + B*z
        arbitrage = -pyo.value(model.obj) + cost_of_battery * z_cap_store
        data.append([benef, arbitrage, z_cap_store])
    df = pd.DataFrame(columns=["depreciated profits", "arbitrage only", "z_cap"], data=data)
    df["node index"] = range(1, d.shape[0])
    df = df[["node index", "depreciated profits", "arbitrage only", "z_cap"]]

    print(df.round(3).to_latex())

