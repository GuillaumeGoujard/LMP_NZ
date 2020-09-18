import lmpnz.JupyterRessources.dataPackage as dataPackage
from lmpnz.GuillaumeExample import economic_dispatch
import lmpnz.JupyterRessources.NZGridVizualization as nzgridvisu
import numpy as np
import pandas as pd
from lmpnz import stored_path
from lmpnz.Network.PriceBids.Generator.Generator import Generator
from lmpnz.Network.Topology.Topology import Topology as topology
import lmpnz.Network.Topology.Topology as Topology
from lmpnz.JupyterRessources import averagecongestion
from lmpnz.JupyterRessources import tablescore

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
    d = dataPackage.tweak_d(d, load_factor=1.3, index_to_tweak=10, load_factor_for_node=12.7) #1.3, 12.7
    plotprog.plot_nodal_demand(d)
    print("load historical data loaded")

    """
    Adjust the bids so that the simulated LMPs are the actual LMPs
    """
    lambdas, gammas = baseline_prices(day, Horizon_T, H, h, Mn, b, P_max, P_min, d)

    actual_lmps = pd.read_csv(stored_path.main_path + "/data/historicaLMPs/Wholesale_price_trends_20200825111248.csv")
    actual_system_lambda = actual_lmps["Price ($/MWh)"].values
    offset = actual_system_lambda - gammas

    b = b+offset
    _, congestion_charge_node = averagecongestion.get_average_congestion_charge(node="MDN1101")
    congestion_charge_node = congestion_charge_node.sort_values(by="Trading_period")
    b[-1] = congestion_charge_node["Price"].values

    lambdas, gammas = baseline_prices(day, Horizon_T, H, h, Mn, b, P_max, P_min, d)
    plotprog.plot_lambdas_gammas(lambdas, gammas)

    """
    table score
    """
    # df = compute_table_score(lambdas, 180, *[d, b, P_max, P_min, H, h, Mn], power_rate=2, list_of_nodes=[3,4,10])

    # print(df.round(1).to_latex(index=False))

    args = [d, b, P_max, P_min, H, h, Mn]
    df = tablescore.compute_table_score(lambdas, 180, *args, power_rate=2)
    print(df.round(1).to_latex())

    plotprog.plot_cum_prof(10, lambdas, 100, *args)
    plotprog.plot_norm_2(*args, z_caps=np.linspace(1, 100, 40))

