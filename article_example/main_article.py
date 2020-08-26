import article_example.dataPackage as dataPackage
from main.GuillaumeExample import price_making_algorithm
from main.GuillaumeExample import economic_dispatch
from main.GuillaumeExample import price_taking_algorithm
from article_example.tablescore import compute_table_score
from article_example.averagecongestion import get_average_congestion_charge
from scipy import optimize
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

    """
    Input used for the article
    """
    day = 2
    Horizon_T = 48
    # root = 2.45
    select_investment_horizon = False


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
    Get the bid matrices from historical bid 
    """
    # Average congestion on MDN
    node_name = "MDN"
    AMB_network.add_generator(Generator("swing_generator", node_name, 0, 0, Pmax=200, Pmin=0,
                                        marginal_cost=[0, 0]))

    b, P_max, P_min = dataPackage.get_producers_matrices(AMB_network, day, Horizon_T, random_a=False)
    print("historical bids loaded")

    """
    Load now the topology of generators
    """
    Mn = AMB_network.Mn

    """
    Get LMP without Battery and plot them
    """
    d = dataPackage.get_load_matrix(AMB_network, day, Horizon_T)
    d = dataPackage.tweak_d(d, load_factor=1.3, index_to_tweak=10, load_factor_for_node=12.7)
    plotprog.plot_nodal_demand(d)
    print("load historical data loaded")

    """
    Adjust the bids so that the simulated LMPs are the actual LMPs
    """
    lambdas, gammas = baseline_prices(day, Horizon_T, H, h, Mn, b, P_max, P_min, d)

    actual_lmps = pd.read_csv("Wholesale_price_trends_20200825111248.csv")
    actual_system_lambda = actual_lmps["Price ($/MWh)"].values
    offset = actual_system_lambda - gammas

    b = b+offset
    _, congestion_charge_node = get_average_congestion_charge(node = "MDN1101")
    congestion_charge_node = congestion_charge_node.sort_values(by="Trading_period")
    b[-1] = congestion_charge_node["Price"].values

    lambdas, gammas = baseline_prices(day, Horizon_T, H, h, Mn, b, P_max, P_min, d)
    plotprog.plot_lambdas_gammas(lambdas, gammas)

    """
    table score
    """
    df = compute_table_score(lambdas, 180, *[d, b, P_max, P_min, H, h, Mn], power_rate=2)
    print(df.round(1).to_latex(index=False))

    # """
    # Influence investment over capacity
    # """
    # select_investment_horizon = False
    # if select_investment_horizon:
    #     y_s = np.linspace(1,4,10)  # amortize in 10 years
    #     sequence_of_costs = [200 * 1000 / (y * 365) for y in y_s]
    #
    #     sequence_of_costs = np.linspace(200, 100, 10)
    #     caps = {}
    #     for c in sequence_of_costs:
    #         model = price_making_algorithm.run_program(d, b, P_max, P_min, H, h, Mn, i_battery=10,
    #                                                    max_capacity=None, cost_of_battery=int(c), power_rate=2)
    #         caps[c] = pyo.value(model.z_cap)
    #
    #     caps_south = {}
    #     for c in sequence_of_costs:
    #         model = price_making_algorithm.run_program(d, b, P_max, P_min, H, h, Mn, i_battery=1,
    #                                                    max_capacity=None, cost_of_battery=int(c), power_rate=2)
    #         caps_south[c] = pyo.value(model.z_cap)
    #
    # best_c = 182
    # y = (200 * 1000)/(182*365)
    #
    # c = 182
    # model = price_making_algorithm.run_program(d, b, P_max, P_min, H, h, Mn, i_battery=10,
    #                                            max_capacity=None, cost_of_battery=c, power_rate=2)
    # model = price_making_algorithm.run_program(d, b, P_max, P_min, H, h, Mn, i_battery=1,
    #                                            max_capacity=None, cost_of_battery=182)
    # # #empirically select
    # y = 4

    # compute_table_score(0, 0, *[d, b, P_max, P_min, H, h, Mn])
    args = [d, b, P_max, P_min, H, h, Mn]
    df = compute_table_score(lambdas, 180, *args, power_rate=2)
    print(df.round(1).to_latex())

    plotprog.plot_cum_prof(10, lambdas, 35, *args)
    plotprog.plot_norm_2(*args, z_caps=np.linspace(1, 100, 11))

    # """
    # plot 2
    # """
    # extra = False
    # if extra == True:
    #     Horizon_T, day = 48, 2
    #     n = d.shape[0]  # number of nodes
    #
    #     z_cap = 35
    #
    #     print("Launching model...")
    #     model_pm = price_making_algorithm.run_program(d, b, P_max, P_min, H, h, Mn, i_battery=1,
    #                                                   max_capacity=1, cost_of_battery=0, power_rate=2)
    #     # pyo.value(model_pm.obj)
    #     model_pm.z_cap.pprint()
    #
    #     u = [pyo.value(model_pm.u[t]) for t in range(Horizon_T)]  # or do that for array variable
    #     lambda_ = [[pyo.value(model_pm.lambda_[i, t]) for t in range(Horizon_T)] for i in range(d.shape[0])]
    #
    #     df = pd.DataFrame(data=[u, lambda_[10]]).T
    #     df["benefits"] = df[0] * df[1]
    #     df["cumulated profits"] = df["benefits"].cumsum()
    #     df["z"] = [pyo.value(model_pm.z[t]) for t in range(Horizon_T)]
    #
    #     # test = pd.DataFrame(data=[lambdas[10], lambda_[10]])
    #
    #     model_pt = price_taking_algorithm.run_program(lambdas, i_battery=10, cost_of_battery=0, max_capacity=z_cap,
    #                                                   power_rate=2)
    #     df_pt = pd.DataFrame()
    #     # df_pt["u"] = planning
    #     planning = [pyo.value(model_pt.u[t]) for t in range(Horizon_T)]  # planning of push and pull from the network
    #     expected_profits = sum([planning[i] * lambdas[10, i] for i in
    #                             range(Horizon_T)])  # expected profits (lambda cross u with lambdas as exogenous)
    #     df_pt["u"] = planning
    #     df_pt["e_profits"] = [planning[i] * lambdas[10, i] for i in
    #                           range(Horizon_T)]
    #     df_pt["cumulated e profits"] = df_pt["e_profits"].cumsum()
    #
    #     n_lambdas = np.zeros((n, Horizon_T))  # new prices !
    #     for j in range(Horizon_T):
    #         """
    #         Here we sell and buy at 0 (i.e we self-schedule_) the quantity devised in the optimization algorithm
    #         """
    #         model = economic_dispatch.run_ED_1period(d[:, j], b[:, j], P_max[:, j], P_min[:, j], 0, planning[j], H, h, Mn,
    #                                                  i_battery=10)
    #         for k in range(n):
    #             n_lambdas[k, j] = model.dual[model.injection_definition[k]]
    #
    #     # actual_profits = sum([planning[i] * n_lambdas[1, i] for i in range(Horizon_T)])
    #     df_pt["a_profits"] = [planning[i] * n_lambdas[10, i] for i in
    #                           range(Horizon_T)]
    #     df_pt["cumulated a profits"] = df_pt["a_profits"].cumsum()
    #     df_pt["z"] = [pyo.value(model_pt.z[t]) for t in range(Horizon_T)]
    #
    #     plt.title("LMPs on node 10 taker vs maker")
    #     plt.plot(lambdas[10], label=r'$\lambda_{pm}$')
    #     plt.plot(n_lambdas[10], label=r'$\lambda_{pt}$')
    #     plt.legend()
    #     plt.ylabel('\$')
    #     plt.xlabel("Time [trading periods]")
    #     plt.grid("True")
    #     plt.show()
    #
    #     fig, axs = plt.subplots(2, sharex=True, figsize=(12, 8), dpi=80, facecolor='w', edgecolor='k')
    #     plt.subplots_adjust(hspace=.0)
    #     fs = 20
    #
    #     axs[0].plot(df["cumulated profits"], color="black", marker="x", label="Cumulated profits - price maker")
    #     axs[0].plot(df_pt["cumulated e profits"], color="red", marker="o", label="Expected Cumulated profits - price taker")
    #     axs[0].plot(df_pt["cumulated a profits"], color="green", marker="*",
    #                 label="Actuals Cumulated profits - price maker")
    #     axs[0].set_ylabel('\$', fontsize=fs)
    #     # axs[0].set_ylim(bottom=0)
    #     axs[0].set_title('Cumulated profits and Cleared volumes, Maker vs Taker \n Baseline model, Sept. 2nd, 2019',
    #                      fontsize=fs)
    #     axs[0].grid()
    #     axs[0].legend()
    #
    #     axs[1].plot(df["z"], color="black", marker="x", linewidth=3, label="SOC - price maker")
    #     axs[1].plot(df_pt["z"], color="green", marker="x", linestyle="dashed", label="SOC - price taker")
    #     # for i, y_arr, label in zip(range(1, 20), Y[1:, :], Nodes[1:].tolist()):
    #     #     if (label == 'MDN') | (label == 'HEN'):
    #     #         axs[1].plot(y_arr, label=f'{i} : {label}', linewidth=5)
    #     #     else:
    #     #         axs[1].plot(y_arr, label=f'{i} : {label}')
    #
    #     axs[1].legend()
    #     axs[1].set_ylabel('MWh', fontsize=fs)
    #     axs[1].set_xlabel('Time [h]', fontsize=fs)
    #     plt.xticks(range(0, 48, 2), range(24))
    #     axs[1].set_xlim([0, 48])
    #     axs[1].grid()
    #     plt.show()

    """
    Find lambdas for the day (they will be deemed exogenous)
    """

    # P_max[-1] = 500
    # z_caps = np.linspace(1, 100, 11)
    # n = d.shape[0]
    # lambdas = np.zeros((n, Horizon_T))
    # for j in range(Horizon_T):
    #     """
    #     Here is a new optimization framework which is rigoursely the same as devised in the algorithm,
    #     WARNING this is just for time period j.
    #
    #     We input the c and q, the price and quantity offered by the battery. Here 0,0 because we want the LMPs
    #     without the battery
    #     """
    #     c = 0
    #     q = 0
    #     model = economic_dispatch.run_ED_1period(d[:, j], b[:, j], P_max[:, j], P_min[:, j], c, q, H, h, Mn)
    #     for k in range(n):
    #         lambdas[k, j] = model.dual[model.injection_definition[k]]  # here we store the dual variables of the injection definition constraint
    #
    # lambda_base = lambdas[10]
    #
    # lambdas_cap = []
    #  # = [5, 10] + list(np.linspace(15, 200, 5))
    # for z_cap in z_caps:
    #     print(z_cap)
    #     i_Battery = 10
    #     model = price_taking_algorithm.run_program(lambdas, i_battery=i_Battery, cost_of_battery=0, max_capacity=z_cap)
    #     planning = [pyo.value(model.u[t]) for t in range(Horizon_T)]  # planning of push and pull from the network
    #     expected_profits = sum([planning[i] * lambdas[1, i] for i in
    #                             range(Horizon_T)])  # expected profits (lambda cross u with lambdas as exogenous)
    #
    #     n_lambdas = np.zeros((n, Horizon_T))  # new prices !
    #     for j in range(Horizon_T):
    #         """
    #         Here we sell and buy at 0 (i.e we self-schedule_) the quantity devised in the optimization algorithm
    #         """
    #         model = economic_dispatch.run_ED_1period(d[:, j], b[:, j], P_max[:, j], P_min[:, j], 0, planning[j], H, h,
    #                                                  Mn,
    #                                                  i_battery=i_Battery)
    #         for k in range(n):
    #             n_lambdas[k, j] = model.dual[model.injection_definition[k]]
    #
    #     actual_profits = sum([planning[i] * n_lambdas[1, i] for i in range(Horizon_T)])
    #     lambdas_pt = n_lambdas[i_Battery]
    #     lambdas_cap.append(lambdas_pt)
    #
    # lambdas_cap_pm = []
    # # z_caps_pm = np.linspace(1, 500, 10)
    # z_caps_pm = z_caps
    # for z_cap in z_caps_pm:
    #     print(z_cap)
    #     i_Battery = 10
    #     model = price_making_algorithm.run_program(d, b, P_max, P_min, H, h, Mn, i_battery=i_Battery,
    #                                                max_capacity=z_cap, cost_of_battery=0)
    #     lambda_ = [[pyo.value(model.lambda_[i, t]) for t in range(Horizon_T)] for i in range(d.shape[0])]
    #     lambdas_pt = lambda_[i_Battery]
    #     lambdas_cap_pm.append(lambdas_pt)
    #
    # # z_caps = [z_caps[3]] + z_caps[:3] + z_caps[4:]
    # norm = []
    # norm_pm = []
    # for n_l in lambdas_cap:
    #     norm.append(np.sqrt(sum(np.array(lambda_base - n_l) ** 2)))
    # for n_l in lambdas_cap_pm:
    #     norm_pm.append(np.sqrt(sum(np.array(lambda_base - n_l) ** 2)))
    #
    # # norm = [norm[3]] + norm[:3] + norm[4:]
    #
    # plt.figure(figsize=(10, 6))
    # plt.plot(z_caps, norm, marker="x", linestyle="dashed",
    #          label=r'price taker : $||\lambda^{bl}_{10} - \lambda^{pt}_{10}||_2$')
    # plt.plot(z_caps_pm, norm_pm, marker="o", label=r'price maker : $||\lambda^{bl}_{10} - \lambda^{pm}_{10}||_2$')
    # plt.title("Price taker vs maker strategy influence on LMP at node 10 in function of installed capacity")
    # plt.xlabel(r'$z^{cap}$', fontsize=17)
    # # plt.axhline(y=0, label="no difference in prices", color="black", linestyle="dotted")
    # # plt.ylabel("Norm 2 Difference between lmp without and with battery")
    # plt.legend(fontsize=17)
    # plt.show()