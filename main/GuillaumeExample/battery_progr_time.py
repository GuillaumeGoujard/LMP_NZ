import numpy as np
import pyomo.environ as pyo

b = np.array([0, 25, 50, 200])  # 4 generators
a = np.array([0, 0, 0, 0])  # 4 generators
P_max = np.array([1, 3, 3, 5])
d = 6  # demand of 6 MW
L = 10000
Horizon_T = 12
Battery_Horizon = Horizon_T+1
d = np.array([6] * Horizon_T)
interesting_D = [6]*Horizon_T
interesting_D[Horizon_T//2] = 1
d = np.array(interesting_D)
# d = np.array([6])
# d = np.array([6,10,3])

print("bids = ", b)
print("Pmax = ", P_max)
print("Demand = ", d)
# print("Price and quantity for battery :", c_u, q_u)


z_min = 1
z_max = 5
A = np.zeros((2*Battery_Horizon, Battery_Horizon))
for t in range(Battery_Horizon):
    A[2 * t, t] = 1
    A[2 * t + 1, t] = -1
z_bar = np.array([z_max, -z_min] * Battery_Horizon)

E = np.zeros((Horizon_T, Battery_Horizon))
for t in range(0, Horizon_T):
    E[t, t+1] = 1
    E[t, t] = -1
I_tilde = np.eye(Horizon_T)


# I_tilde[Horizon_T - 1, Horizon_T - 1] = 0

model = pyo.ConcreteModel(name="feasibility_analysis")
# model.productors_index = range(len(b))
model.prod_times_index = pyo.Set(initialize=list((i, j) for i in range(len(b)) for j in range(Horizon_T)))
model.time_index = range(Horizon_T)
model.battery_index = range(Battery_Horizon)

model.g_t = pyo.Var(model.prod_times_index, domain=pyo.NonNegativeReals)

model.z = pyo.Var(model.battery_index, domain=pyo.NonNegativeReals)
model.u = pyo.Var(model.time_index, domain=pyo.Reals)
model.q_u = pyo.Var(model.time_index, domain=pyo.NonNegativeReals)
model.q_u_test = pyo.Var(domain=pyo.NonNegativeReals)
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
model.r_u = pyo.Var(model.time_index, domain=pyo.Binary)

cost_battery = 1
def obj_func(model):
    S = 0
    for t in range(Horizon_T):
        S += d[t] * model.lambda_[t]
        for i in range(len(b)):
            S += - b[i] * model.g_t[i, t] - P_max[i] * model.sigma[i, t]
    return -S + cost_battery*model.q_u_test


model.obj = pyo.Objective(rule=obj_func)
# model.obj = pyo.Objective(rule=lambda model : 1)


# obj_func = lambda model: -((pyo.summation(b, model.g_t) + pyo.summation(np.array([d]), model.lambda_) + P_max@model.sigma))

def equality(model, t):
    return sum(model.g_t[:, t]) - d[t] + model.u[t] == 0


model.balance_constraint = pyo.Constraint(model.time_index, rule=equality)


# def at_least_u(model, t):
#     return model.u[t] >= 0.1
#
#
# model.at_least_u = pyo.Constraint(model.time_index, rule=at_least_u)


def prod_constraint(model, i, t):
    return model.g_t[i, t] <= P_max[i]


def prod_constraint_u(model, t):
    return model.u[t] <= model.q_u_test  #model.q_u[t]


model.bid_prod = pyo.Constraint(model.prod_times_index, rule=prod_constraint)
model.bid_prod_u = pyo.Constraint(model.time_index, rule=prod_constraint_u)


def constraint2(model, i, t):
    return b[i] - model.lambda_[t] + model.sigma[i, t] + model.mu[i, t] == 0


def constraint2_u(model, t):
    return model.c_u[t] - model.lambda_[t] + model.sigma_u[t] + model.mu_u[t] == 0


model.dual_balance_constraint = pyo.Constraint(model.prod_times_index, rule=constraint2)
model.dual_balance_constraint_u = pyo.Constraint(model.time_index, rule=constraint2_u)


def constraint3_u(model, t):
    return model.c_u[t] <= model.lambda_[t] - 0.1


# model.u_taken = pyo.Constraint(model.time_index, rule=constraint3_u)


def sigma_g_cstr1(model, i, t):
    return model.sigma[i, t] <= (1 - model.r_sigma_g[i, t]) * L


def sigma_g_cstr2(model, i, t):
    return P_max[i] - model.g_t[i, t] <= model.r_sigma_g[i, t] * L


def sigma_g_cstr1_u(model, t):
    return model.sigma_u[t] <= (1 - model.r_sigma_g_u[t]) * L


def sigma_g_cstr2_u(model, t):
    return model.q_u_test - model.u[t] <= model.r_sigma_g_u[t] * L #model.q_u[t]


model.slack_bid1 = pyo.Constraint(model.prod_times_index, rule=sigma_g_cstr1)
model.slack_bid2 = pyo.Constraint(model.prod_times_index, rule=sigma_g_cstr2)
model.slack_bid1_u = pyo.Constraint(model.time_index, rule=sigma_g_cstr1_u)
model.slack_bid2_u = pyo.Constraint(model.time_index, rule=sigma_g_cstr2_u)


def sigma_cstrmu_q(model, i, t):
    return model.g_t[i, t] <= model.r_g_t[i, t] * L


def sigma_cstrmu(model, i, t):
    return -model.mu[i, t] <= (1 - model.r_g_t[i, t]) * L


def sigma_cstrmu_qu(model, t):
    return model.u[t] <= model.r_g_t_u[t] * L


def sigma_cstrmu_u(model, t):
    return -model.mu_u[t] <= (1 - model.r_g_t_u[t]) * L


model.slack_pos1 = pyo.Constraint(model.prod_times_index, rule=sigma_cstrmu_q)
model.slack_pos2 = pyo.Constraint(model.prod_times_index, rule=sigma_cstrmu)
model.slack_pos1_u = pyo.Constraint(model.time_index, rule=sigma_cstrmu_qu)
model.slack_pos2_u = pyo.Constraint(model.time_index, rule=sigma_cstrmu_u)

model.A = pyo.RangeSet(0, 2*Battery_Horizon-1)
def battery_states_limits(model, a):
    S = 0
    for i in range(Battery_Horizon):
        S += A[a, i] * model.z[i]
    if a % 2 == 0:
        return S <= model.q_u_test
    else:
        return S <= z_bar[a]

def battery_states_update(model, t):
    S = 0
    for i in range(Battery_Horizon):
        S += E[t,i]*model.z[i]
    for i in range(Horizon_T):
        S += I_tilde[t, i] * model.u[i]
    return S == 0

def initial_state(model):
    return model.z[0] == model.q_u_test

def final_state(model):
    return model.z[Battery_Horizon-1] == model.q_u_test

model.battery_states_limits = pyo.Constraint(model.A, rule=battery_states_limits)
model.battery_states_update = pyo.Constraint(model.time_index, rule=battery_states_update)
model.initial_state = pyo.Constraint(rule=initial_state)
model.final_state = pyo.Constraint(rule=final_state)

# def non_negativity_u(model, t):
#     return model.u[t] >= -(1-model.r_u[t])*L
# def non_negativity_c_u(model, t):
#     return model.c_u[t] <= model.r_u[t]*L
#
# model.non_negativity_u = pyo.Constraint(model.time_index, rule=non_negativity_u)
# model.non_negativity_c_u = pyo.Constraint(model.time_index, rule=non_negativity_c_u)

model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT_EXPORT)

solver = pyo.SolverFactory('gurobi')
res = solver.solve(model)
model.pprint()
print("\n___ OBJ ____" )
print(pyo.value(model.obj))
#
# results = [pyo.value(model.g_t[i]) for i in model.productors_index]
# LMPs = [pyo.value(model.lambda_[0])]
# print("Value of u = {}, for a cost of {} and quantity of {}".format(pyo.value(model.u), pyo.value(model.c_u),
#                                                                     pyo.value(model.q_u)))
# print("Dispatch is {} ".format(results))
# print("For LMP : {}".format(LMPs))

# z = np.array([1])
# S = 0
# a=0
# for i in range(Horizon_T):
#     S += A[a, i] * z[i]
# z_bar

# AC = lambda P : 15/P + 0.7 + (0.04/3)*P
# MC = lambda P : 0.7 + (0.08)*P
# P = np.linspace(0,100)
# plt.plot(P, AC(P), label="average OC")
# plt.plot(P, MC(P), label="marginal OC")
# plt.xlabel("Power MWh")
# plt.ylabel("Cost USD/MWh")
# plt.show()
#
# Power = [101, 102, 100, 95, 90, 85, 80, 0, 25, 50, 1]
# Cost = [5910.3, 5922, 5898, 5841, 5784, 5728, 5673, 4918, 5129, 5363, 4926]
# plt.scatter(Power, Cost)
# plt.xlabel("Generator 2 output")
# plt.ylabel("System cost")
# plt.show()
#
# Power = [20, 30 , 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 220, 270]
# Cost = [6410, 6327, 6250, 6178, 6111, 6050, 5994, 5943, 5897, 5857, 5822, 5793, 5768, 5749, 5735,
#         5727, 5724, 5726, 5733, 5764, 5952]
# plt.scatter(Power, Cost)
# plt.xlabel("Generator 2 output")
# plt.ylabel("System cost")
# plt.show()
#
# x = np.arange(1.5, 1.82, 0.02)
# y = np.array([16.9, 17.02, 17.10, 17.21, 17.27, 17.37, 17.46, 17.53, 17.64, 17.74, 17.83,
#           18.83, 21.80, 25.02, 28.02, 31.05, 34.34])
# plt.plot(x,np.log(y), marker="o", linestyle="dashed")
# plt.axvline(x=1.72, color="red")
# plt.xlabel("Load Scalar value")
# plt.ylabel("Log(MP_5)")
# plt.show()