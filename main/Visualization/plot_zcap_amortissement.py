import matplotlib.pyplot as plt

fs = 15
fig_zcap, ax_zcap = plt.subplots(figsize=(15, 8), dpi=80, facecolor='w', edgecolor='k')

ax_zcap.plot(range(1,10),zcap_per_node, color = 'b')
ax_zcap.set_ylabel('$z_{cap}^\star$ (in MW)', fontsize = fs, color = 'b')

plt.setp(ax_zcap, xticks = range(1,10), xticklabels = range(1,10))
plt.title('$z_{cap}^\star$ for each battery cost at node 10', fontsize = fs)
ax_zcap.set_xlabel('Amortissement (in years)', fontsize = fs)
