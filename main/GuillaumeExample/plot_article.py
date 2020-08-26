import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyomo.environ as pyo

"""
Runnable example :

1 node, 1 time period Economic Dispatch
"""
def create_and_solve_simple_model_with_battery(a, d, c_u, q_u, P_max):
    model = pyo.ConcreteModel(name="with battery")
    model.productors_index = range(len(a))
    model.q = pyo.Var(model.productors_index, domain=pyo.NonNegativeReals)
    model.u = pyo.Var(domain=pyo.NonNegativeReals)

    obj_func = lambda model : (pyo.summation(a, model.q) + c_u*model.u)

    def equality(model, d):
        return pyo.summation(model.q) + model.u - d == 0

    def prod_constraint(model, i):
        return model.q[i] <= P_max[i]

    def battery_constraint(model):
        return model.u <= q_u

    model.balance_constraint = pyo.Constraint(rule=lambda model : equality(model, d))
    model.production_constraint = pyo.Constraint(model.productors_index, rule=prod_constraint)
    model.battery = pyo.Constraint(rule=battery_constraint)
    model.obj = pyo.Objective(rule=obj_func)
    # Export and import floating point data
    model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT_EXPORT)

    solver = pyo.SolverFactory('gurobi')
    solver.solve(model, tee=True)
    results = [pyo.value(model.q[i]) for i in model.productors_index]
    return model


"""
Plot the results for 1 node
"""

def plot_results_1_node(clearing_price, b, a, P_max, d, batteries=None, last_clearing_price=50):
    if batteries is not None:
        assert len(batteries) == 2
        c_u, u = batteries
    max_range = max(P_max) if batteries is None else max(max(P_max), u)
    productor_bids = [[a[i]*x + b[i] if x < P_max[i] else np.nan for x in range(max_range) ] for i in range(len(a))]
    productor_bids[0][1] = 0
    if batteries is not None:
        productor_bids += [[c_u if x < u else np.nan for x in range(max_range) ]]
    productor_bids = np.array(productor_bids).reshape(-1)
    if batteries is not None:
        place_battery_bids = np.argwhere(np.argsort(productor_bids) >= (max_range)*len(b))
    sorted_bids = np.sort(productor_bids)

    df = pd.DataFrame(index=np.arange(0, len(sorted_bids)), data= sorted_bids, columns=["cumulated_bids"])
    df = df.dropna()
    df["clearing_price"] = clearing_price

    plt.figure(figsize=(8,6))

    plt.axhline(clearing_price, label="price w battery", linewidth=2, linestyle="--", color="green")
    if batteries is not None:
        index_battery = [iu[0] for iu in place_battery_bids if iu[0] <= len(df.index) ]
        to_plot = list(df["cumulated_bids"].iloc[index_battery])
        plt.step([index_battery[0] - 1] + index_battery, [to_plot[0]] + to_plot, marker="x", markersize=10, color="red",
                 linewidth=2,
                 label='battery bids')
        df1 = df.iloc[:index_battery[0]]
        df2 = df.iloc[index_battery[-1]:]
        df2.iloc[0]["cumulated_bids"] = df2.iloc[1]["cumulated_bids"]
        dfs = [df1, df2]

    labels = ['cumulated bids', None]
    for i, df in enumerate(dfs):
        plt.step(df.index, list(df["cumulated_bids"]), color="blue", label=labels[i], marker='o', )
        plt.plot(df.index, list(df["cumulated_bids"]), 'C0o', label=None, alpha=0.5)
    plt.axvline(x=d, color="purple", linestyle='--', label="demand")
    plt.plot([df1.index[-1], df1.index[-1]], [to_plot[0], df1.iloc[-1]["cumulated_bids"]], label=None, color="blue")
    plt.plot([df2.index[0], df2.index[0]], [to_plot[-1], df2.iloc[0]["cumulated_bids"]], label=None, color="blue")
    plt.axhline(last_clearing_price, label="price w/o battery", linewidth=1, linestyle='dashdot', color="black")
    # plt.legend([["battery bids"], ["cumulated bids"], [], ["test"]])
    plt.legend(prop={'size': 17})
    plt.ylabel("USD/MWh")
    plt.title("Clearing price and bids for a trading period")
    plt.xlabel("Cumulated bids (MWh)")

    plt.show()
    return True


"""
Optimal price-maker battery for one node
"""
def optimal_batter_progr_in_f_time(b, P_max, d):
    L = 10000
    # d=6
    print("bids = ", b)
    print("Pmax = ", P_max)
    print("Demand = ", d)
    # print("Price and quantity for battery :", c_u, q_u)

    Horizon_T = 2
    d = np.array([6]*Horizon_T)

    z_min = 1
    z_max = 5
    A = np.zeros((Horizon_T*2,Horizon_T))
    for t in range(Horizon_T):
        A[2*t,t] = 1
        A[2 * t +1, t] = -1
    z_bar = np.array([z_max, -z_min]*Horizon_T)

    E = np.zeros((Horizon_T, Horizon_T))
    for t in range(1, Horizon_T):
        E[t,t] = 1
        E[t, t-1] = -1

    I_tilde = np.eye(Horizon_T)
    I_tilde[Horizon_T-1, Horizon_T-1] = 0


    model = pyo.ConcreteModel(name="feasibility_analysis")
    model.productors_index = range(len(b))
    model.prod_times_index = pyo.Set(initialize=list((i,j) for i in range(len(b)) for j in range(Horizon_T)))

    model.time_index = range(Horizon_T)
    model.g_t = pyo.Var(model.prod_times_index, domain=pyo.NonNegativeReals)

    model.z = pyo.Var(model.time_index, domain=pyo.NonNegativeReals)
    model.u = pyo.Var(model.time_index, domain=pyo.NonNegativeReals)
    model.q_u = pyo.Var(model.time_index, domain=pyo.NonNegativeReals)
    model.c_u = pyo.Var(model.time_index, domain=pyo.NonNegativeReals)

    model.lambda_ = pyo.Var(model.time_index, domain=pyo.Reals)
    model.sigma = pyo.Var(model.prod_times_index, domain=pyo.NonNegativeReals)
    model.mu = pyo.Var(model.prod_times_index, domain=pyo.NonPositiveReals)
    model.sigma_u = pyo.Var(model.time_index, domain=pyo.NonNegativeReals)
    model.mu_u = pyo.Var(model.time_index, domain=pyo.NonPositiveReals)

    model.r_sigma_g = pyo.Var(model.prod_times_index, domain=pyo.Binary)
    model.r_g_t = pyo.Var(model.prod_times_index, domain=pyo.Binary)
    model.r_sigma_g_u = pyo.Var(model.time_index, domain=pyo.Binary)
    model.r_g_t_u = pyo.Var(model.time_index, domain=pyo.Binary)

    def obj_func(model):
        S = 0
        for t in range(Horizon_T):
            for i in range(len(b)):
                S += b[i]*model.g_t[i,t] - d[t]*model.lambda_[t] + P_max[i]*model.sigma[i,t]
        return S
    model.obj = pyo.Objective(rule=obj_func)

    # obj_func = lambda model: -((pyo.summation(b, model.g_t) + pyo.summation(np.array([d]), model.lambda_) + P_max@model.sigma))

    def equality(model, i, t):
        return sum(model.g_t[:,t]) - d[t] + model.u[t] == 0
    model.balance_constraint = pyo.Constraint(model.prod_times_index, rule=equality)

    def at_least_u(model, t):
        return model.u[t] >= 0.1
    model.at_least_u = pyo.Constraint(model.time_index, rule=at_least_u)

    def prod_constraint(model, i, t):
        return model.g_t[i,t] <= P_max[i]
    def prod_constraint_u(model, t):
        return model.u[t] <= model.q_u[t]
    model.bid_prod = pyo.Constraint(model.prod_times_index, rule=prod_constraint)
    model.bid_prod_u = pyo.Constraint(model.time_index, rule=prod_constraint_u)

    def constraint2(model, i, t):
        l = model.lambda_[0]
        return b[i] - l + model.sigma[i,t] + model.mu[i,t] == 0
    def constraint2_u(model, t):
        l = model.lambda_[0]
        return model.c_u[t] - l + model.sigma_u[t] + model.mu_u[t] == 0
    model.dual_balance_constraint = pyo.Constraint(model.prod_times_index, rule=constraint2)
    model.dual_balance_constraint_u = pyo.Constraint(model.time_index, rule=constraint2_u)

    def sigma_g_cstr1(model, i, t):
        return model.sigma[i,t] <= (1 - model.r_sigma_g[i,t]) * L
    def sigma_g_cstr2(model, i, t):
        return P_max[i] - model.g_t[i,t] <= model.r_sigma_g[i,t] * L
    def sigma_g_cstr1_u(model, t):
        return model.sigma_u[t] <= (1 - model.r_sigma_g_u[t]) * L
    def sigma_g_cstr2_u(model):
        return model.q_u[t] - model.u[t] <= model.r_sigma_g_u[t] * L
    model.slack_bid1 = pyo.Constraint(model.prod_times_index, rule=sigma_g_cstr1)
    model.slack_bid2 = pyo.Constraint(model.prod_times_index, rule=sigma_g_cstr2)
    model.slack_bid1_u = pyo.Constraint(model.time_index,rule=sigma_g_cstr1_u)
    model.slack_bid2_u = pyo.Constraint(model.time_index,rule=sigma_g_cstr2_u)

    def sigma_cstrmu_q(model, i, t):
        return model.g_t[i, t] <= model.r_g_t[i,t] * L
    def sigma_cstrmu(model, i, t):
        return -model.mu[i, t] <= (1 - model.r_g_t[i,t]) * L
    def sigma_cstrmu_qu(model, t):
        return model.u[t] <= model.r_g_t_u[t] * L
    def sigma_cstrmu_u(model, t):
        return -model.mu_u[t] <= (1 - model.r_g_t_u[t]) * L
    model.slack_pos1 = pyo.Constraint(model.prod_times_index, rule=sigma_cstrmu_q)
    model.slack_pos2 = pyo.Constraint(model.prod_times_index, rule=sigma_cstrmu)
    model.slack_pos1_u = pyo.Constraint(model.time_index, rule=sigma_cstrmu_qu)
    model.slack_pos2_u = pyo.Constraint(model.time_index, rule=sigma_cstrmu_u)

    def battery_states_limits(model):
        return A@model.z <= z_bar
    def battery_states_update(model):
        return E@model.z + I_tilde@model.u == 0

    model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT_EXPORT)

    solver = pyo.SolverFactory('gurobi')
    res = solver.solve(model)


    return model



def obj_f_linear():
    b = np.array([0, 20, 40, 50, 100])  # 4 generators
    a = np.array([0, 0, 0, 0, 0])  # 4 generators
    P_max = np.array([1, 2, 3, 3, 1])
    d = 9
    c_u = 50
    u = 1
    clearing_price = 50

    b = np.array([20, 100])  # 4 generators
    a = np.array([0, 0])  # 4 generators
    P_max = np.array([5, 1])
    d = 6
    c_u = 50
    u = 3
    clearing_price = 50

    max_range = max(max(P_max), u)
    productor_bids = [[a[i] * x + b[i] if x < P_max[i] else np.nan for x in range(max_range)] for i in range(len(a))]
    productor_bids[0][1] = 0
    productor_bids += [[c_u if x < u else np.nan for x in range(max_range)]]
    productor_bids = np.array(productor_bids).reshape(-1)
    place_battery_bids = np.argwhere(np.argsort(productor_bids) >= (max_range) * len(b))
    sorted_bids = np.sort(productor_bids)
    df = pd.DataFrame(index=np.arange(0, len(sorted_bids)), data=sorted_bids, columns=["cumulated_bids"])
    df = df.dropna()
    df["clearing_price"] = clearing_price
    df = df.iloc[1:]

    plt.figure(figsize=(7, 6))
    bars = ('A', 'B', 'C', 'D', 'E')
    # Choose the position of each barplots on the x-axis (space=1,4,3,1)
    y_pos = [0.5, 2, 3+1.5, 7, 8.5]
    width = [1, 2, 3, 2, 1]
    height = [0,20,40,50,50]

    y_pos = [2.5, 5]
    width = [3, 2]
    height = [20, 50]
    # Create bars
    plt.bar(y_pos, height, width=width, color=('grey', 'red'),  edgecolor='blue')
    # Create names on the x-axis
    # plt.xticks(y_pos, bars)
    # Show graphic
    plt.step(df.index, df["cumulated_bids"], label='cumulated bids')
    plt.plot(df.index, df["cumulated_bids"], 'C0o', alpha=0.5)
    index_battery = [iu[0] for iu in place_battery_bids if iu[0] <= len(df.index)]
    to_plot = list(df["cumulated_bids"].iloc[index_battery])
    plt.step([4, 5, 6], [50,50, 50], color="red", label='battery bids')

    plt.plot(df.index, df["clearing_price"], label=r'$\lambda$', linestyle=(0, (4, 2)),  color="black")
    plt.axvline(x=d, color="orange", label=r'$d$', linestyle=(0, (4, 2)))
    plt.legend()
    plt.ylabel("USD/MWh")
    plt.title("Clearing price and bids for a trading period")
    plt.xlabel("Cumulated bids (MWh)")
    plt.ylim([0,105])

    plt.show()



if __name__ == '__main__':
    """
    bidding curve for each of the 4 generators is a*x + b
    """
    b = np.array([0, 20, 40, 50, 70])  # 4 generators
    a = np.array([0, 0, 0, 0,  0])  # 4 generators
    P_max = np.array([1, 2, 3, 3, 1])

    b = np.array([0, 30, 50])  # 4 generators
    a = np.array([0, 0, 0])  # 4 generators
    P_max = np.array([1, 2, 3])
    d = 4  # demand of 6 MWh

    # """
    # Create and solve model
    # """
    # model = create_and_solve_simple_model_with_battery(b, d, 0, 0, P_max)
    # clearing_price = model.dual[model.balance_constraint]
    #
    # """
    # Plot results
    # """
    # plot_results_1_node(clearing_price, b, a, P_max, d)

    """
    Price-taker battery self-schedules 6
    """
    c_u, q_u = 0, 3
    u = q_u
    model = create_and_solve_simple_model_with_battery(b, d, c_u, q_u, P_max)
    clearing_price = model.dual[model.balance_constraint]
    plot_results_1_node(clearing_price, b, a, P_max, d, batteries=(c_u, q_u))

    model = create_and_solve_simple_model_with_battery(b, d, 29.99, 3, P_max)
    clearing_price = model.dual[model.balance_constraint]
    plot_results_1_node(clearing_price, b, a, P_max, d, batteries=(29.99, 3))

    #
    # """
    # Find best u, c_u
    # """
    # model = optimal_batter_progr(b, P_max, d)
    # c_u, q_u = pyo.value(model.c_u), pyo.value(model.q_u)
    # c_u = 49.5
    # # c_u = 48
    # q_u = 6
    #
    # model = create_and_solve_simple_model_with_battery(b, d, c_u, q_u, P_max)
    # clearing_price = model.dual[model.balance_constraint]
    #
    # """
    # Plot results
    # """
    # plot_results_1_node(clearing_price, b, a, P_max, d, batteries=(c_u, q_u))



