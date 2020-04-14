from datetime import datetime as datetime
from typing import List, Tuple
import numpy as np
import pandas as pd

class Load:
    def __init__(self, name:str, node_name:str, index:int, type:str, constant_demand=None):
        self.name = name
        self.node_name = node_name
        self.index = index
        self.load_data = None
        if type == "dummy":
            self.d = constant_demand

    def add_load_data(self, load_data: pd.DataFrame):
        """

        :param load_data: dataframe with timestamp !
        :return:
        """
        self.load_data = load_data
        return True


    def return_d(self, daterange: List[datetime]) -> np.array:
        """

        :param daterange: list of range of dates
        :return:
        """
        return self.d

