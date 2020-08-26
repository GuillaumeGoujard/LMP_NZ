import matplotlib.pyplot as plt
from main.GuillaumeExample import price_making_algorithm
from main.GuillaumeExample import economic_dispatch
from main.GuillaumeExample import price_taking_algorithm
import numpy as np
import pandas as pd
import pyomo.environ as pyo


def plot_nodal_demand(d):
    x_ = [0.5*i for i in range(d.shape[1])]
    for i in range(d.shape[0]):
        plt.plot(x_, d[i], label=str(i))
    plt.title("NZ Loads for 19 nodes, for september 2nd 2019 ")
    plt.xlabel("Time (hour)")
    plt.ylabel("Load (MWh)")
    plt.legend()
    plt.show()


def plot_lambdas_gammas(lambdas, gammas):
    fig, axs = plt.subplots(2, sharex=True, figsize=(15, 10), dpi=80, facecolor='w', edgecolor='k')
    plt.subplots_adjust(hspace=.0)
    fs = 20

    axs[0].plot(gammas)
    axs[0].set_ylabel('Average price $\gamma$ [\$/MW]', fontsize=fs)
    # axs[0].set_ylim(bottom=0)
    axs[0].set_title('Market Prices, Sept. 2nd, 2019', fontsize=fs)
    # axs[0].grid()

    for i in range(1, 20):
        if i in [3, 4, 12, 11, 16, 19]:
            axs[1].plot(lambdas[i] - gammas, label=f'Node {i}', alpha=0.5)
        elif i ==10:
            axs[1].plot(lambdas[i] - gammas, label=f'Node {i}', color="black", linewidth=3)
        elif i == 1:
            axs[1].plot(lambdas[i] - gammas, label=f'Uncongested Node', linewidth=3)

    axs[1].legend()
    axs[1].set_ylabel('$\lambda - \gamma$ [\$/MW]', fontsize=fs)
    axs[1].set_xlabel('Time [h]', fontsize=fs)
    plt.xticks(range(0, 48, 2), range(24))
    axs[1].set_xlim([0, 48])
    # axs[1].grid()
    plt.legend(prop={'size': 17})
    plt.show()


def plot_cum_prof(battery_index, lambdas, z_cap, *args):
    # battery_index = 12
    # z_cap = 45
    d, b, P_max, P_min, H, h, Mn = args

    n, Horizon_T = d.shape

    model_pm = price_making_algorithm.run_program(d, b, P_max, P_min, H, h, Mn, i_battery=battery_index,
                                                  max_capacity=z_cap, cost_of_battery=0, power_rate=2)

    u = [pyo.value(model_pm.u[t]) for t in range(Horizon_T)]  # or do that for array variable
    lambda_ = [[pyo.value(model_pm.lambda_[i, t]) for t in range(Horizon_T)] for i in range(d.shape[0])]

    df = pd.DataFrame(data=[u, lambda_[battery_index]]).T
    df["benefits"] = df[0] * df[1]
    df["cumulated profits"] = df["benefits"].cumsum()
    df["z"] = [pyo.value(model_pm.z[t]) for t in range(Horizon_T)]

    # test = pd.DataFrame(data=[lambdas[10], lambda_[10]])

    model_pt = price_taking_algorithm.run_program(lambdas, i_battery=battery_index, cost_of_battery=0, max_capacity=z_cap,
                                                  power_rate=2)
    df_pt = pd.DataFrame()
    # df_pt["u"] = planning
    planning = [pyo.value(model_pt.u[t]) for t in range(Horizon_T)]  # planning of push and pull from the network
    expected_profits = sum([planning[i] * lambdas[battery_index, i] for i in
                            range(Horizon_T)])  # expected profits (lambda cross u with lambdas as exogenous)
    df_pt["u"] = planning
    df_pt["e_profits"] = [planning[i] * lambdas[battery_index, i] for i in
                          range(Horizon_T)]
    df_pt["cumulated e profits"] = df_pt["e_profits"].cumsum()

    n_lambdas = np.zeros((n, Horizon_T))  # new prices !
    for j in range(Horizon_T):
        """
        Here we sell and buy at 0 (i.e we self-schedule_) the quantity devised in the optimization algorithm
        """
        model = economic_dispatch.run_ED_1period(d[:, j], b[:, j], P_max[:, j], P_min[:, j], 0, planning[j], H, h, Mn,
                                                 i_battery=battery_index)
        for k in range(n):
            n_lambdas[k, j] = model.dual[model.injection_definition[k]]

    # actual_profits = sum([planning[i] * n_lambdas[1, i] for i in range(Horizon_T)])
    df_pt["a_profits"] = [planning[i] * n_lambdas[battery_index, i] for i in
                          range(Horizon_T)]
    df_pt["cumulated a profits"] = df_pt["a_profits"].cumsum()
    df_pt["z"] = [pyo.value(model_pt.z[t]) for t in range(Horizon_T)]

    plt.title("LMPs on node 10 taker vs maker")
    plt.plot(lambdas[10], label=r'$\lambda_{pm}$')
    plt.plot(n_lambdas[10], label=r'$\lambda_{pt}$')
    plt.legend(prop={'size': 17})
    plt.ylabel('\$')
    plt.xlabel("Time [trading periods]")
    plt.grid("True")
    plt.show()

    fig, axs = plt.subplots(2, sharex=True, figsize=(12, 10), dpi=80, facecolor='w', edgecolor='k')
    plt.subplots_adjust(hspace=.0)
    fs = 20

    axs[0].plot(df["cumulated profits"], color="black", marker="x", label="Cumulated profits - price maker")
    axs[0].plot(df_pt["cumulated e profits"], color="green", marker="o", label="Expected Cumulated profits - price taker")
    axs[0].plot(df_pt["cumulated a profits"], color="red", marker="*",
                label="Actuals Cumulated profits - price taker")
    axs[0].set_ylabel('\$', fontsize=fs)
    # axs[0].set_ylim(bottom=0)
    axs[0].set_title('Cumulated profits and SOC, Maker vs Taker',
                     fontsize=fs)
    # axs[0].grid()
    axs[0].legend(prop={'size': 17})

    axs[1].plot(df["z"], color="black", marker="x", linewidth=3, label="SOC - price maker")
    axs[1].plot(df_pt["z"], color="green", marker="x", linestyle="dashed", label="SOC - price taker")
    # for i, y_arr, label in zip(range(1, 20), Y[1:, :], Nodes[1:].tolist()):
    #     if (label == 'MDN') | (label == 'HEN'):
    #         axs[1].plot(y_arr, label=f'{i} : {label}', linewidth=5)
    #     else:
    #         axs[1].plot(y_arr, label=f'{i} : {label}')

    axs[1].legend(prop={'size': 17})
    axs[1].set_ylabel('MWh', fontsize=fs)
    axs[1].set_xlabel('Time [h]', fontsize=fs)
    plt.xticks(range(0, 48, 2), range(24))
    axs[1].set_xlim([0, 48])
    # axs[1].grid()
    plt.show()


def plot_norm_2(*args, z_caps = np.linspace(1, 100, 11)):
    d, b, P_max, P_min, H, h, Mn = args
    n, Horizon_T = d.shape
    lambdas = np.zeros((n, Horizon_T))
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
            lambdas[k, j] = model.dual[model.injection_definition[
                k]]  # here we store the dual variables of the injection definition constraint

    lambda_base = lambdas[10]

    lambdas_cap = []
    # = [5, 10] + list(np.linspace(15, 200, 5))
    for z_cap in z_caps:
        print(z_cap)
        i_Battery = 10
        model = price_taking_algorithm.run_program(lambdas, i_battery=i_Battery, cost_of_battery=0, max_capacity=z_cap,
                                                   power_rate=2)
        planning = [pyo.value(model.u[t]) for t in range(Horizon_T)]  # planning of push and pull from the network
        expected_profits = sum([planning[i] * lambdas[1, i] for i in
                                range(Horizon_T)])  # expected profits (lambda cross u with lambdas as exogenous)

        n_lambdas = np.zeros((n, Horizon_T))  # new prices !
        for j in range(Horizon_T):
            """
            Here we sell and buy at 0 (i.e we self-schedule_) the quantity devised in the optimization algorithm
            """
            model = economic_dispatch.run_ED_1period(d[:, j], b[:, j], P_max[:, j], P_min[:, j], 0, planning[j], H, h,
                                                     Mn,
                                                     i_battery=i_Battery)
            for k in range(n):
                n_lambdas[k, j] = model.dual[model.injection_definition[k]]

        actual_profits = sum([planning[i] * n_lambdas[1, i] for i in range(Horizon_T)])
        lambdas_pt = n_lambdas[i_Battery]
        lambdas_cap.append(lambdas_pt)

    lambdas_cap_pm = []
    # z_caps_pm = np.linspace(1, 500, 10)
    z_caps_pm = z_caps
    for z_cap in z_caps_pm:
        print(z_cap)
        i_Battery = 10
        model = price_making_algorithm.run_program(d, b, P_max, P_min, H, h, Mn, i_battery=i_Battery,
                                                   max_capacity=z_cap, cost_of_battery=0, power_rate=2)
        lambda_ = [[pyo.value(model.lambda_[i, t]) for t in range(Horizon_T)] for i in range(d.shape[0])]
        lambdas_pt = lambda_[i_Battery]
        lambdas_cap_pm.append(lambdas_pt)

    # z_caps = [z_caps[3]] + z_caps[:3] + z_caps[4:]
    norm = []
    norm_pm = []
    for n_l in lambdas_cap:
        norm.append(np.sqrt(sum(np.array(lambda_base - n_l) ** 2)))
    for n_l in lambdas_cap_pm:
        norm_pm.append(np.sqrt(sum(np.array(lambda_base - n_l) ** 2)))

    # norm = [norm[3]] + norm[:3] + norm[4:]

    plt.figure(figsize=(10, 6))
    plt.plot(z_caps, norm, marker="x", linestyle="dashed",
             label=r'price taker : $||\lambda^{bl}_{10} - \lambda^{pt}_{10}||_2$')
    plt.plot(z_caps_pm, norm_pm, marker="o", label=r'price maker : $||\lambda^{bl}_{10} - \lambda^{pm}_{10}||_2$')
    plt.title("Price taker vs maker strategy influence on LMP at node 10 in function of installed capacity")
    plt.xlabel(r'$z^{cap}$', fontsize=17)
    # plt.axhline(y=0, label="no difference in prices", color="black", linestyle="dotted")
    # plt.ylabel("Norm 2 Difference between lmp without and with battery")
    plt.legend(fontsize=17)
    plt.show()