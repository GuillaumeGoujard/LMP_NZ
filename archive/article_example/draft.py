# average_congestion = get_average_congestion_charge(node="HEN0331")


# average_congestion = 20
"""
Get LMP without Battery and plot them
"""
# if root is None:
#     def f(lf, node_to_tweak=12, average_congestion=average_congestion):
#         d = dataPackage.get_load_matrix(AMB_network, day, Horizon_T)
#         d = dataPackage.tweak_d(d, load_factor=1, index_to_tweak=node_to_tweak, load_factor_for_node=lf)
#         lambdas, gammas = baseline_prices(day, Horizon_T, H, h, Mn, b, P_max, P_min, d)
#         return np.max(abs(lambdas[node_to_tweak] - gammas)) - average_congestion
#
#
#     hist_ = []
#     for l in np.linspace(1.8, 2.5, 11):
#         d = dataPackage.get_load_matrix(AMB_network, day, Horizon_T)
#         d = dataPackage.tweak_d(d, load_factor=1, index_to_tweak=12, load_factor_for_node=l)
#         # d[3] = d[3] + 1490 - max(d[3])
#         plotprog.plot_nodal_demand(d)
#         lambdas, gammas = baseline_prices(day, Horizon_T, H, h, Mn, b, P_max, P_min, d)
#         t = abs(lambdas[3] - gammas)
#         hist_.append(t[t>1])
#
#
#
#
#     f(1, node_to_tweak=12, average_congestion=20)
#     root = optimize.bisect(lambda x: f(x, node_to_tweak=12, average_congestion=average_congestion), 1, 2.6, xtol=0.05)

congestions = {}
for t in range(d.shape[1]):
    congestions[t] = []
    for i in range(d.shape[0]):
        if (abs(lambdas[i,t]-gammas[t]) > 5):
            congestions[t].append((i, abs(lambdas[i,t]-gammas[t])))
            # print("time {} node {}".format(t, i))
