import numpy as np
import pyomo.environ as pyo

L = 10000
def run_program(d, realLMPs, P_max, H, h, Mn):

    """
    Defining spatial and temporal constants
    """
    Horizon_T = d.shape[1]
    n_nodes = d.shape[0]
    n_lines = H.shape[0]
    n_generators = len(P_max)


    """
    Defining optimization variables
    """
    model = pyo.ConcreteModel(name="feasibility_analysis")

    # Indexes over the optimization variables
    model.prod_times_index = pyo.Set(initialize=list((i, j) for i in range(n_generators) for j in range(Horizon_T)))
    model.bid_index = range(n_generators)
    model.time_index = range(Horizon_T)
    model.nodal_index = pyo.Set(initialize=list((i, j) for i in range(n_nodes) for j in range(Horizon_T)))
    model.beta_index = pyo.Set(initialize=list((i, j) for i in range(n_lines) for j in range(Horizon_T)))
    model.H_index =  pyo.Set(initialize=list((i, j) for i in range(n_lines) for j in range(H.shape[1])))

    """
    H parameter
    """
    model.H = pyo.Param(model.H_index, initialize=lambda model, i, j: H_init(model, i, j, H), mutable=True)

    """
    E.D primal variables
    """
    model.g_t = pyo.Var(model.prod_times_index, domain=pyo.NonNegativeReals)
    model.p_t = pyo.Var(model.nodal_index, domain=pyo.Reals)

    model.b = pyo.Var(model.bid_index, domain=pyo.Reals)

    """
    E.D dual variables
    """
    model.lambda_ = pyo.Var(model.nodal_index, domain=pyo.Reals)
    model.gamma_ = pyo.Var(model.time_index, domain=pyo.Reals)
    model.beta = pyo.Var(model.beta_index, domain=pyo.NonNegativeReals)
    model.sigma = pyo.Var(model.prod_times_index, domain=pyo.NonNegativeReals)
    model.mu = pyo.Var(model.prod_times_index, domain=pyo.NonPositiveReals)

    """
    Binary variables for slack constraints
    """
    model.r_beta_ = pyo.Var(model.beta_index, domain=pyo.Binary)
    model.r_sigma_g = pyo.Var(model.prod_times_index, domain=pyo.Binary)
    model.r_g_t = pyo.Var(model.prod_times_index, domain=pyo.Binary)

    """
    Define objective
    """
    model.obj = pyo.Objective(rule=lambda model : obj_func(model, Horizon_T, n_nodes, realLMPs))

    """
    Injection feasibility constraints
    """
    model.injection_definition = pyo.Constraint(model.nodal_index, rule=lambda model, j, t :
                                                            pt_definition(model, j, t, n_nodes, Mn, d, n_generators))
    model.injection_balance = pyo.Constraint(model.time_index, rule=lambda model, t : injection_balance(model, t, n_nodes))
    model.line_constraints = pyo.Constraint(model.beta_index, rule=lambda model, j,
                                                                           t : line_constraints(model, j, t, n_nodes, h))

    """
    Upper bounds on bids
    """
    model.upper_bound_bid_generators = pyo.Constraint(model.prod_times_index, rule=lambda model, i, t:
                                                                                    prod_constraint(model, i, t, P_max))

    """
    Cost and dual prices for generators
    """
    model.dual_generator_constraint = pyo.Constraint(model.prod_times_index, rule=lambda model, i, t:
                                                                        generator_price(model, i, t, n_nodes, Mn))
    model.LMPs = pyo.Constraint(model.nodal_index, rule=lambda model, i, t: LMP_s(model, i, t, n_nodes, H))



    """
    Slack constraints
    """
    model.beta_cstr1 = pyo.Constraint(model.beta_index, rule=beta_cstr1)
    model.beta_cstr2 = pyo.Constraint(model.beta_index, rule=lambda model, j, t : beta_cstr2(model, j, t, n_nodes, h))
    model.sigma_g_cstr1 = pyo.Constraint(model.prod_times_index, rule=sigma_g_cstr1)
    model.sigma_g_cstr2 = pyo.Constraint(model.prod_times_index, rule=lambda model, i, t :sigma_g_cstr2(model, i, t, P_max))
    model.slack_pos1 = pyo.Constraint(model.prod_times_index, rule=sigma_cstrmu_q)
    model.slack_pos2 = pyo.Constraint(model.prod_times_index, rule=sigma_cstrmu)

    """
    Solve and store
    """
    model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT_EXPORT)

    solver = pyo.SolverFactory('gurobi')
    solver.options['TimeLimit'] = 60
    solver.options['mipgap'] = 0.1
    solver.solve(model, tee=True)
    # res = solver.solve(model)
    return model


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


def pt_definition(model, j, t, n_nodes, Mn, d, n_generators):
    S = 0
    # for i in range(n_nodes):
    for b_ in range(n_generators):
        S += Mn[j,b_]*model.g_t[b_, t]
    return model.p_t[j,t] - S + d[j, t] == 0


def prod_constraint(model, i, t, P_max):
    return model.g_t[i, t] <= P_max[i]


def prod_constraint_u(model, t):
    return model.u[t] <= model.q_u[t]


def generator_price(model, i, t, n_nodes, Mn):
    S = 0
    for j in range(n_nodes):
        S += Mn.T[i,j]*model.lambda_[j, t]
    return model.b[i] - S + model.sigma[i, t] + model.mu[i, t] == 0


def battery_price(model, t, n_nodes, Mu):
    S = 0
    for j in range(n_nodes):
        S += Mu.T[0][j] * model.lambda_[j, t]
    return model.c_u[t] - S + model.sigma_u[t] + model.mu_u[t] == 0


def LMP_s(model, i, t, n_nodes, H):
    S = 0
    for j in range(2*n_nodes):
        S += H.T[i,j] * model.beta[j, t]
    return model.gamma_[t] + S - model.lambda_[i,t] == 0


def gamma_cstr1(model, t):
    return model.gamma_[t] <= (1 - model.r_gamma_[t]) * L

def gamma_cstr2(model, t, n_nodes):
    S = 0
    for j in range(n_nodes):
        S += model.p_t[j, t]
    return S <= model.r_gamma_[t]* L


def beta_cstr1(model, i, t):
    return model.beta[i,t] <= (1-model.r_beta_[i,t]) * 10000

def beta_cstr2(model, j, t, n_nodes, h):
    S = 0
    for i in range(n_nodes):
        S += model.H[j, i] * model.p_t[i, t]
    return h[j] - S <= model.r_beta_[j,t] * 1E7

# def lambda_cstr1(model, i, t):
#     return model.lambda_[i,t] <= (1 - model.r_lambda_[i,t]) * L
#
#
# def lambda_cstr2(model, j, t, n_nodes, n_generators, Mn, Mu, d):
#     S = 0
#     for i in range(n_nodes):
#         for b_ in range(n_generators):
#             S += Mn[i, b_] * model.g_t[b_, t]
#         S += Mu[i][0] * model.u[t]
#     return S - d[j,t] - model.p_t[j,t] <= model.r_lambda_[j,t] * L #model.p_t[j, t] == S - d[j, t]

def sigma_g_cstr1(model, i, t):
    return model.sigma[i, t] <= (1 - model.r_sigma_g[i, t]) * L

def sigma_g_cstr2(model, i, t, P_max):
    return P_max[i] - model.g_t[i, t] <= model.r_sigma_g[i, t] * L

def sigma_g_cstr1_u(model, t):
    return model.sigma_u[t] <= (1 - model.r_sigma_g_u[t]) * L

def sigma_g_cstr2_u(model, t):
    return model.q_u[t] - model.u[t] <= model.r_sigma_g_u[t] * L

def sigma_cstrmu_q(model, i, t):
    return model.g_t[i, t] <= model.r_g_t[i, t] * L


def sigma_cstrmu(model, i, t):
    return -model.mu[i, t] <= (1 - model.r_g_t[i, t]) * L


def sigma_cstrmu_qu(model, t):
    return model.u[t] <= model.r_g_t_u[t] * L


def sigma_cstrmu_u(model, t):
    return -model.mu_u[t] <= (1 - model.r_g_t_u[t]) * L

def battery_bid_cstr(model,t):
    return model.q_u[t] <= model.q_u_test

def battery_states_limits(model, a, Battery_Horizon, A, z_bar):
    S = 0
    for i in range(Battery_Horizon):
        S += A[a, i] * model.z[i]
    if a % 2 == 0:
        return S <= model.q_u_test
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


def get_battery_matrices(Battery_Horizon, z_max=10, z_min=1):
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


def obj_func(model, Horizon_T, n_nodes, realLMPs):
    S = 0
    for t in range(Horizon_T):
        for j in range(n_nodes):
            S += (realLMPs[j, t] - model.lambda_[j, t])**2
    return S




if __name__ == '__main__':
    """
    Real test
    """
    from main.Network.Topology.Topology import Topology as top
    import main.Network.PriceBids.Load.Load as ld
    from main.Network.PriceBids.Generator.Generator import Generator
    from main.Network.PriceBids.Load.Load import Load
    from main.GuillaumeExample import LMP
    import pandas as pd
    import stored_path
    import json
    import math

    AMB_network = top(network="ABM")

    """
    Create loads on each node
    """
    Existing_sub_nodes = ld.get_existing_subnodes()
    historical_loads = ld.get_historical_loads()
    Simp_nodes_dict = ld.get_nodes_to_subnodes()
    Simp_nodes_dict["MAN"] = ["MAN2201"]
    Existing_sub_nodes.append("MAN2201")
    nodes_to_index = pd.read_csv(stored_path.main_path + '/data/ABM/ABM_Nodes.csv')
    for i, node in enumerate(AMB_network.names_2_nodes.keys()):
        # print("Load added at node : " + node)
        index = nodes_to_index[nodes_to_index["Node names"] == node]["Node index"].values[0]
        load = Load(node, node, index, type="real_load")
        load.add_load_data(historical_loads, Simp_nodes_dict, Existing_sub_nodes)
        AMB_network.add_load(load)


    file_path = stored_path.main_path + '/data/generators/generator_adjacency_matrix_dict.json'
    with open(file_path) as f:
        data = json.loads(f.read())

    number_of_added_generators = 0
    for name_generator in data.keys():
        L_ = data[name_generator]
        try:
            if type(L_[0]) != float:
                if not math.isnan(L_[-2]):
                    g = Generator(name_generator, L_[0], 0, L_[-1], Pmax=L_[-2], Pmin=L_[-3], marginal_cost=L_[1])
                    AMB_network.add_generator(g)
                    number_of_added_generators += 1
        except:
            pass

    """
    get d_t for day 12 and trading period 1
    """
    Horizon_T = 24
    d = []
    for k,node in enumerate(AMB_network.loads.keys()):
        d.append([])
        for j in range(Horizon_T):
            d[k].append(1000*AMB_network.loads[node][0].return_d(1, j+1))

    d = np.array(d)

    realLMPS = np.empty(d.shape)
    for j in range(Horizon_T):
        test = LMP.get_vector_LMP(1,j+1)
        realLMPS[:,j] = test.reshape(test.shape[0], 1)[:,0]

    # realLMPs

    """
    Add topology specific characteristics
    """
    n_generator = AMB_network.get_number_of_gen()
    b = np.zeros(n_generator)
    P_max = np.zeros(n_generator)
    for node in AMB_network.generators.keys():
        for g in AMB_network.generators[node] :
            b[g.index] = g.a
            P_max[g.index] = g.Pmax

    list_of_generator = {}
    for node in AMB_network.generators.keys():
        for g in AMB_network.generators[node] :
            list_of_generator[g.index] = g
    new_dict = {}
    for key in sorted(list_of_generator.keys()):
        new_dict[key] = list_of_generator[key]

    # P_min, P_max = AMB_network.create_Pmin_Pmax()
    # P_max = P_max.reshape(-1) #*10
    H, h = AMB_network.H, AMB_network.h
    # h = h
    # h = np.array(list(h)[:23] + list(-h)[:23])
    Mn = AMB_network.Mn
    i_battery = 1
    model = run_program(d, realLMPS, P_max, H, h, Mn)
    # model.pprint()
    # print("\n___ OBJ ____")
    # print(pyo.value(model.obj))

    # for v in model.component_objects(pyo.Var, descend_into=True):
    #     v.pprint()


