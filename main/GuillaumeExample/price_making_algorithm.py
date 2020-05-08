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
from main.Network.Topology.Topology import Topology as top

L = 10000
def run_program(d, b, P_max, P_min, H, h, Mn, i_battery=1, max_capacity=None, cost_of_battery=1):
    """
    Defining spatial and temporal constants
    """
    Horizon_T = d.shape[1]
    n_nodes = d.shape[0]
    Battery_Horizon = Horizon_T + 1
    n_generators = b.shape[0]
    n_lines = H.shape[0]

    """
    Battery state equations
    """
    A, z_bar, I_tilde, E = get_battery_matrices(Battery_Horizon, z_max=10, z_min=1)
    Mu = np.zeros(n_nodes)
    Mu[i_battery] = 1

    """
    Defining optimization variables
    """
    model = pyo.ConcreteModel(name="price making algo")

    # Indexes over the optimization variables
    model.prod_times_index = pyo.Set(initialize=list((i, j) for i in range(b.shape[0]) for j in range(Horizon_T)))
    model.time_index = range(Horizon_T)
    model.battery_index = range(Battery_Horizon)
    model.mu_index = range(n_nodes)
    model.nodal_index = pyo.Set(initialize=list((i, j) for i in range(n_nodes) for j in range(Horizon_T)))
    model.beta_index = pyo.Set(initialize=list((i, j) for i in range(n_lines) for j in range(Horizon_T)))
    model.A = pyo.RangeSet(0, 2 * Battery_Horizon - 1)
    model.H_index = pyo.Set(initialize=list((i, j) for i in range(n_lines) for j in range(H.shape[1])))

    """
    H parameter
    """
    model.H = pyo.Param(model.H_index, initialize=lambda model, i, j: H_init(model, i, j, H), mutable=True)

    """
    Battery variables
    """
    model.z = pyo.Var(model.battery_index, domain=pyo.NonNegativeReals)
    model.q_u = pyo.Var(model.time_index, domain=pyo.NonNegativeReals)
    model.z_cap = pyo.Var(domain=pyo.NonNegativeReals) #max capacity
    model.c_u = pyo.Var(model.time_index, domain=pyo.NonNegativeReals)
    model.starting_z = pyo.Var(domain=pyo.NonNegativeReals)

    """
    E.D primal variables
    """
    model.g_t = pyo.Var(model.prod_times_index, domain=pyo.Reals)
    model.p_t = pyo.Var(model.nodal_index, domain=pyo.Reals)
    model.u = pyo.Var(model.time_index, domain=pyo.Reals)

    """
    E.D dual variables
    """
    model.lambda_ = pyo.Var(model.nodal_index, domain=pyo.Reals)
    model.gamma_ = pyo.Var(model.time_index, domain=pyo.Reals)
    model.beta = pyo.Var(model.beta_index, domain=pyo.NonNegativeReals)
    model.sigma = pyo.Var(model.prod_times_index, domain=pyo.NonNegativeReals)
    model.mu = pyo.Var(model.prod_times_index, domain=pyo.NonPositiveReals)
    model.sigma_u = pyo.Var(model.time_index, domain=pyo.NonNegativeReals)
    model.mu_u = pyo.Var(model.time_index, domain=pyo.NonPositiveReals)

    """
    Binary variables for slack constraints
    """
    model.r_beta_ = pyo.Var(model.beta_index, domain=pyo.Binary)
    model.r_sigma_g = pyo.Var(model.prod_times_index, domain=pyo.Binary)
    model.r_g_t = pyo.Var(model.prod_times_index, domain=pyo.Binary)
    model.r_mu_t = pyo.Var(model.prod_times_index, domain=pyo.Binary)
    model.r_sigma_g_u = pyo.Var(model.time_index, domain=pyo.Binary)
    model.r_g_t_u = pyo.Var(model.time_index, domain=pyo.Binary)
    model.r_u = pyo.Var(model.time_index, domain=pyo.Binary)
    model.r_c = pyo.Var(model.time_index, domain=pyo.Binary)

    """
    Define objective
    """
    model.obj = pyo.Objective(rule=lambda model : obj_func(model, Horizon_T, d, b, P_max, P_min, n_lines, h, n_generators, n_nodes,
                                                           cost_of_battery))

    """
    Injection feasibility constraints
    """
    model.injection_definition = pyo.Constraint(model.nodal_index, rule=lambda model, j, t :
                                                            pt_definition(model, j, t, Mn, d, n_generators, Mu))
    model.injection_balance = pyo.Constraint(model.time_index, rule=lambda model, t : injection_balance(model, t, n_nodes))
    model.line_constraints = pyo.Constraint(model.beta_index, rule=lambda model, j,
                                                                           t : line_constraints(model, j, t, n_nodes, h))

    """
    Upper bounds on bids
    """
    model.upper_bound_bid_generators = pyo.Constraint(model.prod_times_index, rule=lambda model, i, t:
                                                                                    prod_constraint(model, i, t, P_max))
    model.upper_bound_bid_battery = pyo.Constraint(model.time_index, rule=prod_constraint_u)
    model.down_bound_bid_generators =  pyo.Constraint(model.prod_times_index, rule=lambda model, i, t:
                                                                                    prod_constraint_min(model, i, t, P_min))

    """
    Cost and dual prices for generators
    """
    model.dual_generator_constraint = pyo.Constraint(model.prod_times_index, rule=lambda model, i, t:
                                                                        generator_price(model, i, t, n_nodes, Mn, b))
    model.dual_battery_constraint = pyo.Constraint(model.time_index, rule=lambda model, t:
                                                                        battery_price(model, t, n_nodes, Mu))
    model.LMPs = pyo.Constraint(model.nodal_index, rule=lambda model, i, t: LMP_s(model, i, t, n_nodes, H))

    """
    bid constraint for battery 
    """
    model.positivity_battery_bid = pyo.Constraint(model.time_index, rule=positivity_battery_bid)
    model.positivity_price_bid = pyo.Constraint(model.time_index, rule=positivity_price_bid)

    """
    Battery states
    """
    if max_capacity is not None:
        model.capacity_equality = pyo.Constraint(rule=lambda model:model.z_cap==max_capacity)
    model.battery_states_limits = pyo.Constraint(model.A,
                                                 rule=lambda model, a: battery_states_limits(model, a, Battery_Horizon,
                                                                                             A, z_bar, z_cap=max_capacity))
    model.battery_states_update = pyo.Constraint(model.time_index,
                                                 rule=lambda model, t : battery_states_update(model, t, Battery_Horizon, E, Horizon_T,
                                                                            I_tilde))
    model.initial_state = pyo.Constraint(rule=initial_state)
    model.final_state = pyo.Constraint(rule=lambda model : final_state(model, Battery_Horizon))
    model.battery_bid_cstr = pyo.Constraint(model.time_index, rule=battery_bid_cstr)
    model.capacity_constraint = pyo.Constraint(rule=battery_capacity_cstr)

    """
    Slack constraints
    """
    model.beta_cstr1 = pyo.Constraint(model.beta_index, rule=beta_cstr1)
    model.beta_cstr2 = pyo.Constraint(model.beta_index, rule=lambda model, j, t : beta_cstr2(model, j, t, n_nodes, h))
    model.sigma_g_cstr1 = pyo.Constraint(model.prod_times_index, rule=sigma_g_cstr1)
    model.sigma_g_cstr2 = pyo.Constraint(model.prod_times_index, rule=lambda model, i, t :sigma_g_cstr2(model, i, t, P_max))
    model.sigma_g_cstr1_u = pyo.Constraint(model.time_index, rule=sigma_g_cstr1_u)
    model.sigma_g_cstr2_u = pyo.Constraint(model.time_index, rule=sigma_g_cstr2_u)
    model.slack_pos1 = pyo.Constraint(model.prod_times_index, rule=lambda model, i, t: sigma_cstrmu_q(model, i, t, P_min))
    model.slack_pos2 = pyo.Constraint(model.prod_times_index, rule=sigma_cstrmu)
    model.slack_pos1_u = pyo.Constraint(model.time_index, rule=sigma_cstrmu_qu)
    model.slack_pos2_u = pyo.Constraint(model.time_index, rule=sigma_cstrmu_u)

    """
    Solve and store
    """
    model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT_EXPORT)

    solver = pyo.SolverFactory('gurobi')
    res = solver.solve(model)
    return model


def positivity_price_bid(model, t):
    return model.c_u[t] <= model.r_c[t] * L


def positivity_battery_bid(model, t):
    return model.q_u[t] >= (model.r_c[t] - 1) * L


def injection_balance(model, t, n_nodes):
    S = 0
    for j in range(n_nodes):
        S += model.p_t[j, t]
    return S == 0

def line_constraints(model, j, t, n_nodes, h):
    S = 0
    for i in range(n_nodes):
        S += model.H[j, i]*model.p_t[i, t]
    return S <= h[j]


def pt_definition(model, j, t, Mn, d, n_generators, Mu):
    S = 0
    for b_ in range(n_generators):
        S += Mn[j,b_]*model.g_t[b_, t]
    S += Mu[j]*model.u[t]
    return model.p_t[j,t] - S + d[j, t] == 0


def prod_constraint(model, i, t, P_max):
    return model.g_t[i, t] <= P_max[i,t]


def prod_constraint_u(model, t):
    return model.u[t] <= model.q_u[t]


def prod_constraint_min(model, i, t, P_min):
    return model.g_t[i, t] >= P_min[i,t]


def generator_price(model, i, t, n_nodes, Mn, b):
    S = 0
    for j in range(n_nodes):
        S += Mn.T[i,j]*model.lambda_[j, t]
    return b[i,t] - S + model.sigma[i, t] + model.mu[i, t] == 0


def battery_price(model, t, n_nodes, Mu):
    S = 0
    for j in range(n_nodes):
        S += Mu[j] * model.lambda_[j, t]
    return model.c_u[t] - S + model.sigma_u[t] + model.mu_u[t] == 0


def LMP_s(model, i, t, n_nodes, H):
    S = 0
    for j in range(2*n_nodes):
        S += H.T[i,j] * model.beta[j, t]
    return model.gamma_[t] + S - model.lambda_[i,t] == 0


def beta_cstr1(model, i, t):
    return model.beta[i,t] <= (1-model.r_beta_[i,t]) * 10000

def beta_cstr2(model, j, t, n_nodes, h):
    S = 0
    for i in range(n_nodes):
        S += model.H[j, i] * model.p_t[i, t]
    return - h[j] + S <= model.r_beta_[j,t] * 10000

def sigma_g_cstr1(model, i, t):
    return model.sigma[i, t] <= (1 - model.r_sigma_g[i, t]) * L

def sigma_g_cstr2(model, i, t, P_max):
    return P_max[i,t] - model.g_t[i, t] <= model.r_sigma_g[i, t] * L


def sigma_g_cstr1_u(model, t):
    return model.sigma_u[t] <= (1 - model.r_sigma_g_u[t]) * L


def sigma_g_cstr2_u(model, t):
    return model.q_u[t] - model.u[t] <= model.r_sigma_g_u[t] * L


def sigma_cstrmu_q(model, i, t, P_min):
    return model.g_t[i, t] - P_min[i,t] <= model.r_g_t[i, t] * L


def sigma_cstrmu(model, i, t):
    return -model.mu[i, t] <= (1 - model.r_g_t[i, t]) * L


def sigma_cstrmu_qu(model, t):
    return model.u[t] <= model.r_g_t_u[t] * L


def sigma_cstrmu_u(model, t):
    return -model.mu_u[t] <= (1 - model.r_g_t_u[t]) * L


def battery_bid_cstr(model,t):
    return model.q_u[t] <= model.z_cap


def battery_capacity_cstr(model):
    return model.z_cap <= 2000


def battery_states_limits(model, a, Battery_Horizon, A, z_bar, z_cap=None):
    S = 0
    for i in range(Battery_Horizon):
        S += A[a, i] * model.z[i]
    if a % 2 == 0:
        if z_cap is not None:
            return S <= z_cap
        else:
            return S <= model.z_cap
    else:
        return S <= z_bar[a]


def battery_states_update(model, t, Battery_Horizon, E, Horizon_T, I_tilde):
    S = 0
    for i in range(Battery_Horizon):
        S += E[t,i]*model.z[i]
    for i in range(Horizon_T):
        S += I_tilde[t, i] * model.u[i]
    return S == 0


def initial_state(model):
    return model.z[0] == model.starting_z


def final_state(model, Battery_Horizon):
    return model.z[Battery_Horizon-1] == model.starting_z


def get_battery_matrices(Battery_Horizon, z_max=10, z_min=0):
    A = np.zeros((2 * Battery_Horizon, Battery_Horizon))
    for t in range(Battery_Horizon):
        A[2 * t, t] = 1
        A[2 * t + 1, t] = -1
    z_bar = np.array([z_max, -z_min] * Battery_Horizon)

    E = np.zeros((Battery_Horizon-1, Battery_Horizon))
    for t in range(0, Battery_Horizon-1):
        E[t, t + 1] = 1
        E[t, t] = -1
    I_tilde = np.eye(Battery_Horizon-1)
    return A, z_bar, I_tilde, E


def H_init(model, i, j, H):
    return H[i,j]


def obj_func(model, Horizon_T, d, b, P_max, P_min, n_lines, h, n_generators, n_nodes, cost_of_battery):
    S = 0
    for t in range(Horizon_T):
        for j in range(n_nodes):
            S += d[j, t] * model.lambda_[j, t]
        for j in range(n_lines):
            S += - h[j] * model.beta[j, t]
        for i in range(n_generators):
            S += -b[i,t] * (model.g_t[i, t]) - P_max[i,t]*model.sigma[i, t] - P_min[i,t]*model.mu[i,t]
    return -S + cost_of_battery * model.z_cap

