import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyomo.environ as pyo

"""
Runnable example :

4 generators, no battery, fixed demand 
"""
def create_and_solve_simple_model(a, d, P_max):
    model = pyo.ConcreteModel(name="with no battery")
    model.productors_index = range(len(a))
    model.q = pyo.Var(model.productors_index, domain=pyo.NonNegativeReals, initialize=1.0)

    obj_func = lambda model : pyo.summation(a,model.q)

    def equality(model, d):
        return pyo.summation(model.q) - d == 0

    def prod_constraint(model, i):
        return model.q[i] <= P_max[i]

    model.balance_constraint = pyo.Constraint(rule=lambda model : equality(model, d))
    model.production_constraint = pyo.Constraint(model.productors_index, rule=prod_constraint)
    model.obj = pyo.Objective(rule=obj_func)
    # Export and import floating point data
    model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT_EXPORT)

    solver = pyo.SolverFactory('gurobi')
    solver.solve(model, tee=True)
    results = [pyo.value(model.q[i]) for i in model.productors_index]
    return model

"""
Runnable example :
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


def feasibility_analysis_with_battery():
    L = 10000
    d=1.1
    print("bids = ", b)
    print("Pmax = ", P_max)
    print("Demand = ", d)
    print("Price and quantity for battery :", c_u, q_u)

    model = pyo.ConcreteModel(name="feasibility_analysis")
    model.productors_index = range(len(b))
    model.g_t = pyo.Var(model.productors_index, domain=pyo.NonNegativeReals)
    model.u = pyo.Var(domain=pyo.NonNegativeReals)

    model.lambda_ = pyo.Var([0], domain=pyo.Reals)
    model.sigma = pyo.Var(model.productors_index, domain=pyo.NonNegativeReals)
    model.mu = pyo.Var(model.productors_index, domain=pyo.NonPositiveReals)
    model.sigma_u = pyo.Var(domain=pyo.NonNegativeReals)
    model.mu_u = pyo.Var(domain=pyo.NonPositiveReals)

    model.r_sigma_g = pyo.Var(model.productors_index, domain=pyo.Binary)
    model.r_g_t = pyo.Var(model.productors_index, domain=pyo.Binary)
    model.r_sigma_g_u = pyo.Var(domain=pyo.Binary)
    model.r_g_t_u = pyo.Var(domain=pyo.Binary)

    obj_func = lambda model: 1

    def equality(model):
        return pyo.summation(model.g_t) - d + model.u == 0
    model.balance_constraint = pyo.Constraint(rule=equality)

    def prod_constraint(model, i):
        return model.g_t[i] <= P_max[i]
    def prod_constraint_u(model):
        return model.u <= q_u
    model.bid_prod = pyo.Constraint(model.productors_index, rule=prod_constraint)
    model.bid_prod_u = pyo.Constraint(rule=prod_constraint_u)

    def constraint2(model, i):
        l = model.lambda_[0]
        return b[i] - l + model.sigma[i] + model.mu[i] == 0
    def constraint2_u(model):
        l = model.lambda_[0]
        return c_u - l + model.sigma_u + model.mu_u == 0
    model.dual_balance_constraint = pyo.Constraint(model.productors_index, rule=constraint2)
    model.dual_balance_constraint_u = pyo.Constraint(rule=constraint2_u)

    def sigma_g_cstr1(model, i):
        return model.sigma[i] <= (1 - model.r_sigma_g[i]) * L
    def sigma_g_cstr2(model, i):
        return P_max[i] - model.g_t[i] <= model.r_sigma_g[i] * L
    def sigma_g_cstr1_u(model):
        return model.sigma_u <= (1 - model.r_sigma_g_u) * L
    def sigma_g_cstr2_u(model):
        return q_u - model.u <= model.r_sigma_g_u * L
    model.slack_bid1 = pyo.Constraint(model.productors_index, rule=sigma_g_cstr1)
    model.slack_bid2 = pyo.Constraint(model.productors_index, rule=sigma_g_cstr2)
    model.slack_bid1_u = pyo.Constraint(rule=sigma_g_cstr1_u)
    model.slack_bid2_u = pyo.Constraint(rule=sigma_g_cstr2_u)

    def sigma_cstrmu_q(model, i):
        return model.g_t[i] <= model.r_g_t[i] * L
    def sigma_cstrmu(model, i):
        return -model.mu[i] <= (1 - model.r_g_t[i]) * L
    def sigma_cstrmu_qu(model):
        return model.u <= model.r_g_t_u * L
    def sigma_cstrmu_u(model):
        return -model.mu_u <= (1 - model.r_g_t_u) * L
    model.slack_pos1 = pyo.Constraint(model.productors_index, rule=sigma_cstrmu_q)
    model.slack_pos2 = pyo.Constraint(model.productors_index, rule=sigma_cstrmu)
    model.slack_pos1_u = pyo.Constraint(rule=sigma_cstrmu_qu)
    model.slack_pos2_u = pyo.Constraint(rule=sigma_cstrmu_u)

    model.obj = pyo.Objective(rule=obj_func)

    model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT_EXPORT)

    solver = pyo.SolverFactory('gurobi')
    res = solver.solve(model)

    results = [pyo.value(model.g_t[i]) for i in model.productors_index]
    LMPs = [pyo.value(model.lambda_[0])]
    # print("Value of u = {} for a cost of {}".format(pyo.value(model.u), pyo.value(model.c_u)))
    # print("Dispatch is {} ".format(results))
    # print("For LMP : {}".format(LMPs))



def feasibility_analysis():
    L = 10000
    d=1.1
    print("bids = ", b)
    print("Pmax = ", P_max)
    print("Demand = ", d)

    model = pyo.ConcreteModel(name="feasibility_analysis")
    model.productors_index = range(len(b))
    model.g_t = pyo.Var(model.productors_index, domain=pyo.NonNegativeReals)

    model.lambda_ = pyo.Var([0], domain=pyo.Reals)
    model.sigma = pyo.Var(model.productors_index, domain=pyo.NonNegativeReals)
    model.mu = pyo.Var(model.productors_index, domain=pyo.NonPositiveReals)

    model.r_sigma_g = pyo.Var(model.productors_index, domain=pyo.Binary)
    model.r_g_t = pyo.Var(model.productors_index, domain=pyo.Binary)

    obj_func = lambda model: 1

    def equality(model):
        return pyo.summation(model.g_t) - d == 0
    model.balance_constraint = pyo.Constraint(rule=equality)

    def prod_constraint(model, i):
        return model.g_t[i] <= P_max[i]
    model.bid_prod = pyo.Constraint(model.productors_index, rule=prod_constraint)
    # model.bid_prod = pyo.Constraint(model.productors_index, rule=pos_prod_constraint)

    def constraint2(model, i):
        l = model.lambda_[0]
        return b[i] - l + model.sigma[i] + model.mu[i] == 0
    model.dual_balance_constraint = pyo.Constraint(model.productors_index, rule=constraint2)

    def sigma_g_cstr1(model, i):
        return model.sigma[i] <= (1 - model.r_sigma_g[i]) * L
    def sigma_g_cstr2(model, i):
        return P_max[i] - model.g_t[i] <= model.r_sigma_g[i] * L
    model.slack_bid1 = pyo.Constraint(model.productors_index, rule=sigma_g_cstr1)
    model.slack_bid2 = pyo.Constraint(model.productors_index, rule=sigma_g_cstr2)

    def sigma_cstrmu_qu(model, i):
        return model.g_t[i] <= model.r_g_t[i] * L
    def sigma_cstrmu_u(model, i):
        return -model.mu[i] <= (1 - model.r_g_t[i]) * L
    model.slack_pos1 = pyo.Constraint(model.productors_index, rule=sigma_cstrmu_qu)
    model.slack_pos2 = pyo.Constraint(model.productors_index, rule=sigma_cstrmu_u)

    model.obj = pyo.Objective(rule=obj_func)

    model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT_EXPORT)

    solver = pyo.SolverFactory('gurobi')
    res = solver.solve(model)

    results = [pyo.value(model.g_t[i]) for i in model.productors_index]
    LMPs = [pyo.value(model.lambda_[0])]
    # print("Value of u = {} for a cost of {}".format(pyo.value(model.u), pyo.value(model.c_u)))
    print("Dispatch is {} ".format(results))
    print("For LMP : {}".format(LMPs))



def optimal_batter_progr(b, P_max, d):
    L = 10000
    # d=6
    print("bids = ", b)
    print("Pmax = ", P_max)
    print("Demand = ", d)
    # print("Price and quantity for battery :", c_u, q_u)

    model = pyo.ConcreteModel(name="feasibility_analysis")
    model.productors_index = range(len(b))
    model.g_t = pyo.Var(model.productors_index, domain=pyo.NonNegativeReals)
    model.u = pyo.Var(domain=pyo.NonNegativeReals)
    model.q_u = pyo.Var(domain=pyo.NonNegativeReals)
    model.c_u = pyo.Var(domain=pyo.NonNegativeReals)

    model.lambda_ = pyo.Var([0], domain=pyo.Reals)
    model.sigma = pyo.Var(model.productors_index, domain=pyo.NonNegativeReals)
    model.mu = pyo.Var(model.productors_index, domain=pyo.NonPositiveReals)
    model.sigma_u = pyo.Var(domain=pyo.NonNegativeReals)
    model.mu_u = pyo.Var(domain=pyo.NonPositiveReals)

    model.r_sigma_g = pyo.Var(model.productors_index, domain=pyo.Binary)
    model.r_g_t = pyo.Var(model.productors_index, domain=pyo.Binary)
    model.r_sigma_g_u = pyo.Var(domain=pyo.Binary)
    model.r_g_t_u = pyo.Var(domain=pyo.Binary)

    obj_func = lambda model: -((pyo.summation(b, model.g_t) + pyo.summation(np.array([d]), model.lambda_) + P_max@model.sigma))

    def equality(model):
        return pyo.summation(model.g_t) - d + model.u == 0
    model.balance_constraint = pyo.Constraint(rule=equality)

    def at_least_u(model):
        return model.u >= 0.1
    model.at_least_u = pyo.Constraint(rule=at_least_u)

    def prod_constraint(model, i):
        return model.g_t[i] <= P_max[i]
    def prod_constraint_u(model):
        return model.u <= model.q_u
    model.bid_prod = pyo.Constraint(model.productors_index, rule=prod_constraint)
    model.bid_prod_u = pyo.Constraint(rule=prod_constraint_u)

    def constraint2(model, i):
        l = model.lambda_[0]
        return b[i] - l + model.sigma[i] + model.mu[i] == 0
    def constraint2_u(model):
        l = model.lambda_[0]
        return model.c_u - l + model.sigma_u + model.mu_u == 0
    model.dual_balance_constraint = pyo.Constraint(model.productors_index, rule=constraint2)
    model.dual_balance_constraint_u = pyo.Constraint(rule=constraint2_u)

    def sigma_g_cstr1(model, i):
        return model.sigma[i] <= (1 - model.r_sigma_g[i]) * L
    def sigma_g_cstr2(model, i):
        return P_max[i] - model.g_t[i] <= model.r_sigma_g[i] * L
    def sigma_g_cstr1_u(model):
        return model.sigma_u <= (1 - model.r_sigma_g_u) * L
    def sigma_g_cstr2_u(model):
        return model.q_u - model.u <= model.r_sigma_g_u * L
    model.slack_bid1 = pyo.Constraint(model.productors_index, rule=sigma_g_cstr1)
    model.slack_bid2 = pyo.Constraint(model.productors_index, rule=sigma_g_cstr2)
    model.slack_bid1_u = pyo.Constraint(rule=sigma_g_cstr1_u)
    model.slack_bid2_u = pyo.Constraint(rule=sigma_g_cstr2_u)

    def sigma_cstrmu_q(model, i):
        return model.g_t[i] <= model.r_g_t[i] * L
    def sigma_cstrmu(model, i):
        return -model.mu[i] <= (1 - model.r_g_t[i]) * L
    def sigma_cstrmu_qu(model):
        return model.u <= model.r_g_t_u * L
    def sigma_cstrmu_u(model):
        return -model.mu_u <= (1 - model.r_g_t_u) * L
    model.slack_pos1 = pyo.Constraint(model.productors_index, rule=sigma_cstrmu_q)
    model.slack_pos2 = pyo.Constraint(model.productors_index, rule=sigma_cstrmu)
    model.slack_pos1_u = pyo.Constraint(rule=sigma_cstrmu_qu)
    model.slack_pos2_u = pyo.Constraint(rule=sigma_cstrmu_u)

    model.obj = pyo.Objective(rule=obj_func)

    model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT_EXPORT)

    solver = pyo.SolverFactory('gurobi')
    res = solver.solve(model)

    results = [pyo.value(model.g_t[i]) for i in model.productors_index]
    LMPs = [pyo.value(model.lambda_[0])]
    print("Value of u = {}, for a cost of {} and quantity of {}".format(pyo.value(model.u), pyo.value(model.c_u), pyo.value(model.q_u)))
    print("Dispatch is {} ".format(results))
    print("For LMP : {}".format(LMPs))
    return model

def optimal_battery_dispatch():
    L = 10000

    model = pyo.ConcreteModel(name="with battery")

    model.productors_index = range(len(b))
    model.q = pyo.Var(model.productors_index, domain=pyo.Reals)
    # model.c_u = pyo.Var(domain=pyo.NonNegativeReals)
    model.u = pyo.Var(domain=pyo.NonNegativeReals)

    model.lambda_ = pyo.Var([0], domain=pyo.Reals)
    model.sigma = pyo.Var(model.productors_index, domain=pyo.NonNegativeReals)
    model.sigma_u = pyo.Var(domain=pyo.NonNegativeReals)
    model.mu = pyo.Var(model.productors_index, domain=pyo.NonPositiveReals)
    model.mu_u = pyo.Var( domain=pyo.NonPositiveReals)

    model.r_sigma_g = pyo.Var(model.productors_index, domain=pyo.Binary)
    model.r_sigma_u = pyo.Var(domain=pyo.Binary)
    model.r_q = pyo.Var(model.productors_index, domain=pyo.Binary)
    model.r_u = pyo.Var(domain=pyo.Binary)


    # obj_func = lambda model: -((pyo.summation(b, model.q) + pyo.summation(np.array([d]), model.lambda_) + P_max@model.sigma))
    obj_func = lambda model: 1

    d=1
    def equality(model):
        return pyo.summation(model.q) + model.u - d == 0

    def prod_constraint(model, i):
        return model.q[i] <= P_max[i]

    def pos_prod_constraint(model, i):
        return model.q[i] >= 0

    def battery_constraint(model):
        return model.u <= q_u

    def constraint2(model, i):
        return b[i] + model.lambda_[0] + model.sigma[i] + model.mu[i] == 0

    def constraint3(model):
        return c_u + model.lambda_[0] + model.sigma_u + model.mu_u == 0

    def sigma_g_cstr1(model, i):
        return model.sigma[i] <= (1- model.r_sigma_g[i])*L

    def sigma_g_cstr2(model, i):
        return model.q[i] - P_max[i] >= model.r_sigma_g[i]*L

    def sigma_u_cstr1(model):
        return model.sigma_u <= (1 - model.r_sigma_u) * L

    def sigma_u_cstr2(model):
        return model.u - q_u >= model.r_sigma_u * L

    def sigma_cstrmu_q(model,i):
        return model.q[i] <= model.r_q[i] * L

    def sigma_cstrmu(model,i):
        return model.mu[i] >= (1-model.r_q[i]) * L

    def sigma_cstrmu_qu(model):
        return model.u <= model.r_u * L

    def sigma_cstrmu_u(model, i):
        return model.mu_u >= (1 - model.r_u) * L


    model.balance_constraint = pyo.Constraint(rule=equality)
    model.production_constraint = pyo.Constraint(model.productors_index, rule=prod_constraint)
    model.constraint2 = pyo.Constraint(model.productors_index, rule=constraint2)
    model.constraint3 = pyo.Constraint(rule=constraint3)
    model.sigma_g_cstr1 = pyo.Constraint(model.productors_index, rule=sigma_g_cstr1)
    model.sigma_g_cstr2 = pyo.Constraint(model.productors_index, rule=sigma_g_cstr2)
    model.sigma_g_cstru = pyo.Constraint(model.productors_index, rule=sigma_cstrmu_qu)
    model.sigma_g_cstrqu = pyo.Constraint(model.productors_index, rule=sigma_cstrmu_u)
    # model.sigma_u_cstr1 = pyo.Constraint(rule=sigma_u_cstr1)
    # model.sigma_u_cstr2 = pyo.Constraint(rule=sigma_u_cstr2)
    model.battery = pyo.Constraint(rule=battery_constraint)

    model.obj = pyo.Objective(rule=obj_func)

    model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT_EXPORT)

    solver = pyo.SolverFactory('gurobi')
    res = solver.solve(model)

    results = [pyo.value(model.q[i]) for i in model.productors_index]
    LMPs = [pyo.value(model.lambda_[0])]
    print("Value of u = {} for a cost of {}".format(pyo.value(model.u), pyo.value(model.c_u)))
    print("Dispatch is {} ".format(results))
    print("For LMP : {}".format(LMPs))

    pyo.value(model.lambda_)










def plot_results_1_node(clearing_price, b, a, P_max, batteries=None):
    if batteries is not None:
        assert len(batteries) == 2
        c_u, u = batteries
    productor_bids = [[a[i]*x + b[i] if x <= P_max[i] else np.nan for x in range(max(P_max)+1) ] for i in range(len(a))]
    if batteries is not None:
        productor_bids += [[c_u if x <= u else np.nan for x in range(max(P_max)+1) ]]
    productor_bids = np.array(productor_bids).reshape(-1)
    if batteries is not None:
        place_battery_bids = np.argwhere(np.argsort(productor_bids) >= (max(P_max)+1)*len(b))
    sorted_bids = np.sort(productor_bids)
    df = pd.DataFrame(index=np.arange(0, len(sorted_bids)), data= sorted_bids, columns=["cumulated_bids"])
    df = df.dropna()
    df["clearing_price"] = clearing_price

    plt.figure(figsize=(8,8))
    plt.step(df.index, df["cumulated_bids"], label='cumulated bids')
    plt.plot(df.index, df["cumulated_bids"], 'C0o', alpha=0.5)
    if batteries is not None:
        index_battery = [iu[0] for iu in place_battery_bids if iu[0] <= len(df.index) ]
        to_plot = list(df["cumulated_bids"].iloc[index_battery])
        plt.step([index_battery[0] - 1] + index_battery, [to_plot[0]] + to_plot, color="red", label='battery bids')

    plt.plot(df.index, df["clearing_price"], label="clearing price", linestyle= (0, (1,3)))
    plt.axvline(x=d, color="red", label="demand", linestyle= (0, (1,3)))
    plt.legend()
    plt.ylabel("USD/MWh")
    plt.title("Clearing price and bids for a trading period")
    plt.xlabel("Cumulated bids (MWh)")

    plt.show()
    return True

if __name__ == '__main__':
    """
    bidding curve for each of the 4 generators is a*x + b
    """
    b = np.array([0, 20, 50, 200])  # 4 generators
    a = np.array([0, 0, 0, 0])  # 4 generators
    P_max = np.array([1, 3, 3, 1])
    d = 6  # demand of 6 MWh

    """
    Create and solve model
    """
    model = create_and_solve_simple_model(b, d, P_max)
    clearing_price = model.dual[model.balance_constraint]

    """
    Plot results
    """
    plot_results_1_node(clearing_price, b, a, P_max)

    """
    Create and solve model with battery
    """
    c_u, q_u = 10, 2
    model = create_and_solve_simple_model_with_battery(b, d, c_u, q_u, P_max)
    clearing_price = model.dual[model.balance_constraint]

    """
    Plot results
    """
    plot_results_1_node(clearing_price, b, a, P_max, batteries=(c_u, q_u))

    """
    Find best u, c_u
    """
    model = optimal_batter_progr(b, P_max, d)
    c_u, q_u = pyo.value(model.c_u), pyo.value(model.q_u)
    c_u = c_u - 0.1
    # c_u = 48
    q_u = 3

    model = create_and_solve_simple_model_with_battery(b, d, c_u, q_u, P_max)
    clearing_price = model.dual[model.balance_constraint]

    """
    Plot results
    """
    plot_results_1_node(clearing_price, b, a, P_max, batteries=(c_u, q_u))