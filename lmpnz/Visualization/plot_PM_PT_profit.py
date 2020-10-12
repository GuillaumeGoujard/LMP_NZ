import matplotlib.pyplot as plt


profit = -np.array(obj_values_PM) + cost_of_battery*np.array(zcap_per_node_PM)

fs = 15
fig_PM_PT, ax_PM_PT1 = plt.subplots(figsize=(15, 8), dpi=80, facecolor='w', edgecolor='k')
ax_PM_PT2 = ax_PM_PT1.twinx()

ax_PM_PT1.bar(range(1,20), profit, color = 'g')
ax_PM_PT1.set_ylabel('Amortized profit [\$]', fontsize = fs, color = 'g')
ax_PM_PT1.set_xlabel('Node index', fontsize = fs)
ax_PM_PT1.set_ylim(bottom = 0)


ax_PM_PT2.plot(range(1,20),zcap_per_node_PM, color = 'b')
ax_PM_PT2.set_ylabel('$z_{cap}^\star$ (in MWh)', fontsize = fs, color = 'b')
ax_PM_PT2.set_ylim(bottom = 0)

plt.setp(ax_PM_PT1, xticks = range(1,20), xticklabels = range(1,20))
plt.title('Benefits and $z_{cap}^\star$ for each node\n 5 year battery, Sept. 2nd, 2019', fontsize = fs)
plt.grid()










# Profits and SOC at node 10
u = np.array([z_df.u]).T
LMP = np.array(LMP_df[[f'{i}' for i in range(48)]][LMP_df.Node == 10]).T
Y = u * LMP

fs = 15
fig, ax = plt.subplots(figsize=(15, 8), dpi=80, facecolor='w', edgecolor='k')
ax1 = ax.twinx()

ax.plot(range(48), Y, color = 'g')
ax.set_ylabel('Profit [\$]', fontsize = fs, color = 'g')
ax.set_xlabel('Timestep [h]', fontsize = fs)
ax.set_ylim([-6000,6000])

ax1.plot(range(48), z_df.z, color = 'b')
ax1.set_ylabel('SOC', fontsize = fs, color = 'b')
ax1.set_ylim([-100,100])

plt.setp(ax, xticks = range(0,48,2), xticklabels = range(24))
plt.title('PM profits and SOC\n 5 year battery, Node 10, Sept. 2nd, 2019', fontsize = fs)
plt.grid()




# PT expected and actual profits and SOC at node 10
u = np.array(planning)
Y_exp = u * lambdas[10]
Y_act = u * n_lambdas[10]

z = np.array([pyo.value(model_PT.z[i]) for i in range(Horizon_T)])

fs = 15
fig, ax = plt.subplots(figsize=(15, 8), dpi=80, facecolor='w', edgecolor='k')
ax1 = ax.twinx()

ax.plot(range(48), Y_exp, color = 'orange', label = 'Expected profit')
ax.plot(range(48), Y_act, color = 'purple', label = 'Actual profit')

ax.set_ylabel('Profit [\$]', fontsize = fs, color = 'k')
ax.set_xlabel('Timestep [h]', fontsize = fs)
ax.set_ylim([-17000,17000])

ax1.plot(range(48), z, color = 'b')
ax1.set_ylabel('SOC', fontsize = fs, color = 'b')
ax1.set_ylim([-100,100])

plt.setp(ax, xticks = range(0,48,2), xticklabels = range(24))
plt.title('PT expected and actual profits and SOC\n 5 year battery, Node 10, Sept. 2nd, 2019', fontsize = fs)
ax.legend(fontsize = fs)
plt.grid()





# LMP comparison
LMP_base = lambdas[10]
LMP_PM = lambdas_PM[10]
    # np.array(LMP_df[[f'{i}' for i in range(48)]][10:11])[0].tolist()
LMP_PT = n_lambdas[10]

fs = 15
plt.figure(figsize=(15, 8), dpi=80, facecolor='w', edgecolor='k')

plt.plot(range(48), LMP_base, color = 'k', label = 'Normal LMP')
plt.plot(range(48), LMP_PM, color = 'green', linestyle='dashed', linewidth = 2, marker = "+",markersize=12, label = 'Price Maker LMP')
plt.plot(range(48), LMP_PT, color = 'purple', linestyle='dashed',linewidth = 2, marker = "o", label = 'Price Taker LMP')
plt.plot(range(48), gamma[0], color = 'orange', linestyle='dashed', label = 'Price Taker LMP')

plt.ylabel('LMP values [\$/MW]', fontsize = fs, color = 'k')
plt.xlabel('Timestep [h]', fontsize = fs)

plt.xticks(range(0,48,2), range(24))
plt.title('LMP values in the base case, with Price-Maker and with Price-taker optimization\n 5 year battery, Node 10, Sept. 2nd, 2019', fontsize = fs)
plt.legend(fontsize = fs)
plt.grid()






# SOC for PM and PT
SOC_PM = np.array([pyo.value(model_PM.z[i]) for i in range(Horizon_T)])
SOC_PT = np.array([pyo.value(model_PT.z[i]) for i in range(Horizon_T)])

fs = 15
plt.figure(figsize=(15, 8), dpi=80, facecolor='w', edgecolor='k')

plt.plot(range(48), SOC_PM, color = 'green', label = 'PM')
plt.plot(range(48), SOC_PT, color = 'purple', label = 'PT')

plt.ylabel('SOC', fontsize = fs, color = 'k')
plt.xlabel('Timestep [h]', fontsize = fs)
plt.ylim([0,100])

plt.xticks(range(0,48,2), range(24))
plt.title('SOC of PM and PT\n 5 year battery, Node 10, Sept. 2nd, 2019', fontsize = fs)
plt.legend(fontsize = fs)
plt.grid()









