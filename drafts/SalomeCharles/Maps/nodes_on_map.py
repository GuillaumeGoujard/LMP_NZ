import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd

dir_path = '/Users/salomeschwarz/Desktop/Moura_295/Project/code/'

df = pd.read_csv(dir_path+'Nodes_NZ.csv', sep=';')
del df['Unnamed: 5']
del df['Unnamed: 6']

# BBox = ((df.Long.min(),   df.Long.max(),
#         df.Lat.min(), df.Lat.max()))
BBox = (166.430217,   178.548968, -47.284921, -34.393350)

nz = plt.imread(dir_path+"nz.png")
fig, ax = plt.subplots()
ax.axis('off')
plt.show()

def plot_first():
    fig, ax = plt.subplots(figsize=(8,8))
    ax.scatter(df.Long, df.Lat*0.99, zorder=1, alpha= 0.5, c='b', s=df['Capacity (MW)'])
    ax.set_title('nodes NZ electric grid for hydropower')
    ax.set_xlim(BBox[0],BBox[1])
    ax.set_ylim(BBox[2],BBox[3])
    ax.imshow(nz, zorder=0, extent = BBox, aspect= '1.25')
    plt.savefig(dir_path+"locations.pdf", dpi=1000, transparent=True)
    plt.show()
