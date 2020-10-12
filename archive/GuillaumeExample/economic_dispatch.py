import numpy as np
import pyomo.environ as pyo


L = 10000
def run_ED_1period(d, b, P_max, P_min, c, q, H, h, Mn, i_battery=1):
    """
    Defining spatial and temporal constants
    """
    n_nodes = d.shape[0]
    n_generators = b.shape[0]
    n_lines = H.shape[0]

    Mu = np.zeros(n_nodes)
    Mu[i_battery] = 1

    """
    Defining optimization variables
    """
    model = pyo.ConcreteModel(name="economic_dispatch")

    # Indexes over the optimization variables
    model.prod_times_index = range(b.shape[0])
    model.mu_index = range(n_nodes)
    model.nodal_index = range(n_nodes)
    model.beta_index = range(n_lines)
    model.H_index = pyo.Set(initialize=list((i, j) for i in range(n_lines) for j in range(H.shape[1])))

    """
    H parameter
    """
    model.H = pyo.Param(model.H_index, initialize=lambda model, i, j: H_init(model, i, j, H), mutable=True)

    """
    E.D primal variables
    """
    model.g_t = pyo.Var(model.prod_times_index, domain=pyo.Reals)
    model.p_t = pyo.Var(model.nodal_index, domain=pyo.Reals)
    model.u_t = pyo.Var([0], domain=pyo.Reals)

    """
    Define objective
    """
    model.obj = pyo.Objective(rule=lambda model : obj_func(model, b, c, n_generators))

    """
    Injection feasibility constraints
    """
    model.injection_definition = pyo.Constraint(model.nodal_index, rule=lambda model, j :
                                                            pt_definition(model, j, Mn, d, n_generators, Mu))
    model.injection_balance = pyo.Constraint(rule=lambda model : injection_balance(model, n_nodes))
    model.line_constraints = pyo.Constraint(model.beta_index, rule=lambda model, j: line_constraints(model, j, n_nodes, h))

    """
    Upper bounds on bids
    """
    model.upper_bound_bid_generators = pyo.Constraint(model.prod_times_index, rule=lambda model, i:
                                                                                    prod_constraint(model, i, P_max))
    model.upper_bound_bid_battery = pyo.Constraint(rule=lambda mode: prod_constraint_u(model, q))
    model.down_bound_bid_generators = pyo.Constraint(model.prod_times_index, rule=lambda model, i:
                                                                                    prod_constraint_min(model, i, P_min))

    """
    Solve and store
    """
    model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT_EXPORT)

    solver = pyo.SolverFactory('gurobi')
    res = solver.solve(model)
    return model




def injection_balance(model, n_nodes):
    S = 0
    for j in range(n_nodes):
        S += model.p_t[j]
    return S == 0

def line_constraints(model, j, n_nodes, h):
    S = 0
    for i in range(n_nodes):
        S += model.H[j, i]*model.p_t[i]
    return S <= h[j]


def pt_definition(model, j, Mn, d, n_generators, Mu):
    S = 0
    for b_ in range(n_generators):
        S += Mn[j,b_]*model.g_t[b_]
    S += Mu[j]*model.u_t[0]
    return -model.p_t[j] + S - d[j] == 0


def prod_constraint(model, i, P_max):
    return model.g_t[i] <= P_max[i]


def prod_constraint_u(model, q):
    return model.u_t[0] <= q


def prod_constraint_min(model, i, P_min):
    return model.g_t[i] >= P_min[i]


def H_init(model, i, j, H):
    return H[i,j]


def obj_func(model, b, c, n_generators):
    S = 0
    for i in range(n_generators):
        S += b[i]*model.g_t[i]
    return S + c*model.u_t[0]

