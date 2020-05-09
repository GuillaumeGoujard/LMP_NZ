import matplotlib.pyplot as plt


profit = -np.array(obj_values_PM) + cost_of_battery*np.array(zcap_per_node_PM)

fs = 15
fig_PM_PT, ax_PM_PT1 = plt.subplots(figsize=(15, 8), dpi=80, facecolor='w', edgecolor='k')
ax_PM_PT2 = ax_PM_PT1.twinx()

ax_PM_PT1.bar(range(1,20), profit, color = 'g')
ax_PM_PT1.set_ylabel('Benefits (in \$)', fontsize = fs, color = 'g')
ax_PM_PT1.set_xlabel('Node index', fontsize = fs)
ax_PM_PT1.set_ylim(bottom = 0)


ax_PM_PT2.plot(range(1,20),zcap_per_node_PM, color = 'b')
ax_PM_PT2.set_ylabel('$z_{cap}^\star$ (in MW)', fontsize = fs, color = 'b')
ax_PM_PT2.set_ylim(bottom = 0)

plt.setp(ax_PM_PT1, xticks = range(1,20), xticklabels = range(1,20))
plt.title('Benefits and $z_{cap}^\star$ for each node', fontsize = fs)
plt.grid()
