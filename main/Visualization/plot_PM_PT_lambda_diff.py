import matplotlib.pyplot as plt
import numpy as np

# Lambda diffs
lambdas
lambdas_PM
lambdas_PT

diff_PM = np.linalg.norm(lambdas - lambdas_PM, axis = 0)
diff_PT = np.linalg.norm(lambdas - lambdas_PT, axis = 0)



fs = 15
plt.figure(figsize=(15, 8), dpi=80, facecolor='w', edgecolor='k')

plt.plot(range(48), diff_PM, color = 'green', label = 'PM')
plt.plot(range(48), diff_PT, color = 'purple', label = 'PT')

plt.ylabel('$||\lambda_p - \lambda||$', fontsize = fs, color = 'k')
plt.xlabel('Timestep [h]', fontsize = fs)
# plt.ylim([0,100])

plt.xticks(range(0,48,2), range(24))
plt.title('Difference between PM/PT LMP and baseline LMP\n 5 year battery on Node 10, Sept. 2nd, 2019', fontsize = fs)
plt.legend(fontsize = fs)
plt.grid()