import numpy as np
import lmpnz.Network.PriceBids.Generator.Generator as gen

def get_load_matrix(AMB_network, day, Horizon_T):
    d = []
    for k, node in enumerate(AMB_network.loads.keys()):
        d.append([])
        for j in range(day * 48, day * 48 + Horizon_T):
            d[k].append(1 * 1000 * AMB_network.loads[node][0].return_d(1 + j // 48, j % 48 + 1))
    d = np.array(d)
    return d


def get_producers_matrices(AMB_network, day, Horizon_T, random_a=True):
    n_generator = AMB_network.get_number_of_gen()
    b = np.zeros((n_generator, Horizon_T))
    P_max = np.zeros((n_generator, Horizon_T))
    P_min = np.zeros((n_generator, Horizon_T))
    for node in AMB_network.generators.keys():
        for g in AMB_network.generators[node]:
            for i in range(Horizon_T):
                if g.name == "swing_generator":
                    pmax, pmin, a = g.Pmax, g.Pmin, g.a
                else:
                    pmax, pmin, a = gen.get_P_min_a(g.name, day, i+1)
                P_max[g.index, i] = pmax
                P_min[g.index, i] = pmin if g.type == "Hydro" else 0
                if random_a:
                    b[g.index, i] = a if a >= 1 else np.random.randint(0, 50)
                else:
                    b[g.index, i] = a
    return b, P_max, P_min


def tweak_d(d, load_factor = 1.3, index_to_tweak = 10, load_factor_for_node=12.1):
    save = d[index_to_tweak].copy()
    save_d = d.copy()
    d[index_to_tweak] = save
    d = load_factor * save_d.copy()
    d[index_to_tweak] = save * load_factor_for_node
    return d
