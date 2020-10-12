from lmpnz.Network.Topology.Topology import Topology
from lmpnz.Network.PriceBids.Load.Load import Load
from lmpnz.Network.PriceBids.Generator.Generator import Generator
import lmpnz.Network.PriceBids.Load.Load as ld

import pandas as pd

"""
Test : Custom network creation
"""
test_node = {
                 "AA1":0,
                 "AA2":0,
                 "AA3":0,
                 "BB1":1,
                 "BB2":1,
             }
top = Topology(f_names_2_nodes = test_node)

"""
Test : One Node network
"""
OneNode_network = Topology(network = "ONDE")

"""
Test : Two Node network
"""
TwoNode_network = Topology(network = "NSNDE")

"""
Test : AMB Network
"""
AMB_network = Topology(network="ABM")

# AMB_network.names_2_nodes

"""
Create two generators and add them to the network
"""
g = Generator("GuillaumeGenerator", "HEN", 0, "dummy", Pmax=20, Pmin=0, marginal_cost=10)
AMB_network.add_generator(g)
a = Generator("AliceGenerator", "HEN", 2, "dummy", Pmax=200, Pmin=0, marginal_cost=200)
AMB_network.add_generator(a)
k = Generator("KieranGenerator", "MAN", 1, "dummy", Pmax=100, Pmin=5, marginal_cost=50)
AMB_network.add_generator(k)

"""
Create loads on each nodes 
"""
Existing_sub_nodes = ld.get_existing_subnodes()
historical_loads = ld.get_historical_loads()
Simp_nodes_dict = ld.get_nodes_to_subnodes()
nodes_to_index = pd.read_csv('data/ABM/ABM_Nodes.csv')
for i, node in enumerate(AMB_network.names_2_nodes.keys()):
    # print("Load added at node : " + node)
    index = nodes_to_index[nodes_to_index["Node names"] == node]["Node index"].values[0]
    load = Load(node, node, index, type="real_load")
    load.add_load_data(historical_loads, Simp_nodes_dict, Existing_sub_nodes)
    AMB_network.add_load(load)

"""
Add topology specific characteristics
"""
AMB_network.create_Pmin_Pmax()
AMB_network.create_Qt_at()