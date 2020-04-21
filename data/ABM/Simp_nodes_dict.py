Simp_nodes = pd.read_csv('data/ABM/ABM_Simplified_network.csv')
Simp_nodes = Simp_nodes.rename(columns={'Swem Node': "Simp_node", ' NZEM Substations that act as Grid Exit Points': 'Orig_node'})
Simp_nodes_dict = {
    key: Simp_nodes[Simp_nodes.Simp_node == key].Orig_node.values[0].split()
    for key in Simp_nodes.Simp_node.values
}