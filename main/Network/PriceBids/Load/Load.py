import pickle

import numpy as np
import pandas as pd

import stored_path

class Load:
    def __init__(self, name:str, node_name:str, index:int, type:str, constant_demand=None):
        self.name = name
        self.node_name = node_name
        self.index = index
        self.load_data = None
        # self.d = None
        # if type == "dummy":
        #     self.d = constant_demand

    def add_load_data(self, load_data: pd.DataFrame, from_nodes_to_subnodes, Existing_sub_nodes):
        """

        :param load_data: dataframe with timestamp !
        :return:
        """
        nodal_loads = pd.DataFrame()
        if self.node_name not in from_nodes_to_subnodes:
            return None
        list_of_subnodes = from_nodes_to_subnodes[self.node_name]
        save_column = load_data[["day", "Trading period"]].drop_duplicates()
        for n in list_of_subnodes:
            if n in Existing_sub_nodes:
                nodal_data = load_data[load_data["Region ID"] == n]
                nodal_loads = pd.concat([nodal_loads, nodal_data['Demand (GWh)']], axis=1)
        if save_column is None:
            return False
        nodal_loads = pd.concat([save_column, nodal_loads.sum(axis=1)], axis=1)
        nodal_loads.columns = ["day", "period", "load"]
        self.load_data = nodal_loads
        return True


    def return_d(self, day, period) -> np.array:
        """

        :param daterange: list of range of dates
        :return:
        """
        if self.load_data is None:
            return 0
        return self.load_data[(self.load_data["day"] == day) & (self.load_data["period"] == period)]["load"].values[0]


def get_historical_loads():
    historical_loads = pd.read_csv(stored_path.main_path + "/data/loads/Grid_demand_trends_20200421102501.csv")
    historical_loads["date"] = pd.to_datetime(historical_loads["Period start"], format="%d/%m/%Y %H:%M:%S")
    historical_loads["day"] = historical_loads["date"].apply(lambda s: s.day)
    historical_loads = historical_loads[['date', 'day', 'Trading period', 'Region ID', 'Demand (GWh)']]
    historical_loads.index = historical_loads['date']
    return historical_loads

def get_nodes_to_subnodes():
    Simp_nodes = pd.read_csv(stored_path.main_path + '/data/ABM/ABM_Simplified_network.csv')
    Simp_nodes = Simp_nodes.rename(
        columns={'Swem Node': "Simp_node", ' NZEM Substations that act as Grid Exit Points': 'Orig_node'})
    Simp_nodes_dict = {
        key: Simp_nodes[Simp_nodes.Simp_node == key].Orig_node.values[0].split()
        for key in Simp_nodes.Simp_node.values
    }

    return Simp_nodes_dict


def get_existing_subnodes():
    with open(stored_path.main_path +'/data/ABM/sub_nodes.txt', "rb") as fp:  # Unpickling
        b = pickle.load(fp)
    return b

