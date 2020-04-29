from datetime import datetime as datetime
from typing import List, Tuple
import numpy as np
import pandas as pd

class Generator:
    def __init__(self, name:str, node_name:str, index:int, type, Pmax=None, Pmin=None, marginal_cost=None):
        self.name = name
        self.node_name = node_name
        self.index = index
        self.type = type
        self.Pmax = Pmax
        self.Pmin = Pmin
        self.a, self.q = marginal_cost[0], marginal_cost[1]
        # self.q, self.a, self.k = None, None, None
        if type == "dummy":
            self.a = marginal_cost
            self.k, self.q = 0, 0



    def fit_curve(self, bid_curves_data, input_data=None):
        """
        Find q, a and k s.t
            => C(g) = 1/2 q g^2 + a*g + (k^Tu)*g
        :param bid_curves_data: for each time period, the observed bidding curve of the generator
        :param input_data: for each time period, the observed vector of input data
        :return: q, a, k that minimizes || \sum_t C(g;q,a,k)_t - Bc_t ||^2 (or another objective function !)
        """
        pass


    def add_input_data(self, input_data: pd.DataFrame):
        """

        :param input_data: dataframe with timestamp !
        :return:
        """
        self.input_data = input_data
        return True


    def return_a_q_k(self, daterange: List[datetime]) -> Tuple[np.array, np.array]:
        """

        :param daterange: list of range of dates for whcih
        :return:
        """
        returned_a = self.a + self.k.T@self.input_data.loc[daterange[0]]
        return self.q, returned_a

