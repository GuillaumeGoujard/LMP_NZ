from lmpnz.GuillaumeExample import price_making_algorithm
from lmpnz.GuillaumeExample import economic_dispatch
from lmpnz.GuillaumeExample import price_taking_algorithm
import numpy as np
import pandas as pd
import pyomo.environ as pyo


def compute_table_score(lambdas, capacity_cost, *args, power_rate=2, list_of_nodes=None):
    data = []
    n = args[0].shape[0]
    Horizon_T = args[0].shape[1]
    d, b, P_max, P_min, H, h, Mn = args
    list_of_nodes = list_of_nodes if list_of_nodes is not None else list(range(1, n))
    for i_battery in list_of_nodes:
        print(i_battery)
        model = price_making_algorithm.run_program(*args, i_battery=i_battery,
                                                   max_capacity=None, cost_of_battery=capacity_cost, power_rate=power_rate)
        z_cap_store = pyo.value(model.z_cap)

        benef = -pyo.value(model.obj)  # model.obj = -lambda*u + B*z
        arbitrage = -pyo.value(model.obj) + capacity_cost * z_cap_store
        """
        test
        """
        model_pt = price_taking_algorithm.run_program(lambdas, i_battery=i_battery, cost_of_battery=0,
                                                      max_capacity=z_cap_store,
                                                      power_rate=2)
        df_pt = pd.DataFrame()
        # df_pt["u"] = planning
        planning = [pyo.value(model_pt.u[t]) for t in
                    range(Horizon_T)]  # planning of push and pull from the network
        expected_profits = sum([planning[i] * lambdas[10, i] for i in
                                range(Horizon_T)])  # expected profits (lambda cross u with lambdas as exogenous)
        df_pt["u"] = planning
        df_pt["e_profits"] = [planning[i] * lambdas[10, i] for i in
                              range(Horizon_T)]
        df_pt["cumulated e profits"] = df_pt["e_profits"].cumsum()

        n_lambdas = np.zeros((n, Horizon_T))  # new prices !
        for j in range(Horizon_T):
            """
            Here we sell and buy at 0 (i.e we self-schedule_) the quantity devised in the optimization algorithm
            """
            model = economic_dispatch.run_ED_1period(d[:, j], b[:, j], P_max[:, j], P_min[:, j], 0, planning[j], H,
                                                     h, Mn,
                                                     i_battery=i_battery)
            for k in range(d.shape[0]):
                n_lambdas[k, j] = model.dual[model.injection_definition[k]]

        actual_profits = sum([planning[i] * n_lambdas[i_battery, i] for i in range(Horizon_T)])

        data.append([z_cap_store, benef, arbitrage, expected_profits, actual_profits])

    df = pd.DataFrame(columns=["z_cap", "depreciated profits", "arbitrage only pm", "expected arbitrage pt",
                               "actual arbitrage pt"], data=data)
    df["unit arbitrage pt"] = df["expected arbitrage pt"]/df["z_cap"]
    df["node index"] = list_of_nodes
    df = df.round(1)
    df = df[["node index"] + list(df.columns)[:-1]]
    return df.T
