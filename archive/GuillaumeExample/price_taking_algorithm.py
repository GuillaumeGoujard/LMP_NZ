import numpy as np
import pyomo.environ as pyo


L = 10000
def run_program(lambdas, i_battery=1, max_capacity=None, final_s=None, cost_of_battery=1, power_rate=1):
    """
    Defining spatial and temporal constants
    """
    Horizon_T = lambdas.shape[1]

    """
    Battery state equations
    """
    Battery_Horizon = Horizon_T + 1
    A, z_bar, I_tilde, E = get_battery_matrices(Battery_Horizon, z_max=10, z_min=0)
    lambda_ = lambdas[i_battery]

    """
    Defining optimization variables
    """
    model = pyo.ConcreteModel(name="price taking algo")

    # Indexes over the optimization variables
    model.time_index = range(Horizon_T)
    model.battery_index = range(Battery_Horizon)
    model.A = pyo.RangeSet(0, 2 * Battery_Horizon - 1)


    """
    Battery variables
    """
    model.z = pyo.Var(model.battery_index, domain=pyo.NonNegativeReals)
    model.u = pyo.Var(model.time_index, domain=pyo.Reals)
    model.z_cap = pyo.Var(domain=pyo.NonNegativeReals) #max capacity
    model.starting_z = pyo.Var(domain=pyo.NonNegativeReals)

    """
    Define objective
    """
    model.obj = pyo.Objective(rule=lambda model : obj_func(model, Horizon_T, lambda_, cost_of_battery))


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
    model.final_state = pyo.Constraint(rule=lambda model : final_state(model, Battery_Horizon, final_state=final_s))
    model.capacity_constraint = pyo.Constraint(rule=battery_capacity_cstr)
    model.ramp_down = pyo.Constraint(model.time_index,
                                                 rule=lambda model, t: ramp_down(model, t, power_rate=power_rate) )
    model.ramp_up = pyo.Constraint(model.time_index,
                                                 rule=lambda model, t: ramp_up(model, t, power_rate=power_rate))

    """
    Solve and store
    """
    model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT_EXPORT)

    solver = pyo.SolverFactory('gurobi')
    res = solver.solve(model)
    return model



def battery_capacity_cstr(model):
    return model.z_cap <= 2000


def ramp_up(model, t, power_rate=1):
    return model.u[t] <= model.z_cap/power_rate

def ramp_down(model, t, power_rate=1):
    return model.u[t] >= -model.z_cap/power_rate

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


def final_state(model, Battery_Horizon, final_state=None):
    if final_state is not None:
        return model.z[Battery_Horizon - 1] == final_state
    else:
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


def obj_func(model, Horizon_T, lambda_, cost_of_battery):
    S = 0
    for t in range(Horizon_T):
        S += -lambda_[t]*model.u[t]
    return S + cost_of_battery * model.z_cap

