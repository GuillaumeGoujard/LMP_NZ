from datetime import datetime as datetime
from typing import List, Tuple
import numpy as np
import pandas as pd
import stored_path
import swifter

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


path = stored_path.main_path + '/drafts/SalomeCharles/Data/'

def create_data_Sept():

    df1 = pd.read_csv(path + '20190901_Offers.csv', parse_dates=['UTCSubmissionTime'])
    df2 = pd.read_csv(path + '20190902_Offers.csv', parse_dates=['UTCSubmissionTime'])
    df3 = pd.read_csv(path + '20190903_Offers.csv', parse_dates=['UTCSubmissionTime'])

    dfs = []
    dfs_ = [df1, df2, df3]
    for i, df in enumerate(dfs_):
        df = df[df['ProductType'] != 'Reserve']

        df['name'] = df['Unit'].astype(str) + '_' + df['PointOfConnection']

        df["date"] = pd.to_datetime(df["TradingDate"], format="%Y-%m-%d")
        df["day"] = df["date"].swifter.apply(lambda s: s.day)

        dfs.append(df)

    df = pd.concat(dfs)

    return df


df = create_data_Sept()

def get_P_min_a(generator, day,  tp):
    df_ = df[(df['name'] == generator)&(df['TradingPeriod'] == tp)&(df['day'] == day)].copy()
    df_ = df_[df_["IsLatest"]=="Y"]
    df_ = df_.sort_values(by='DollarsPerMegawattHour')
    df_['cumsum_MW'] = df_['Megawatt'].cumsum()

    cap = 0.001
    min_power = df_['cumsum_MW'][df_.DollarsPerMegawattHour <= cap]
    P_min = min_power.iloc[-1] if len(min_power) > 0 else 0
    P_max = df_['cumsum_MW'].iloc[-1] if len(df_) > 0 else 0
    new_df = df_[df_.DollarsPerMegawattHour > cap].copy()
    if len(new_df) == 0 or sum(new_df['cumsum_MW']) == 0 or sum(new_df['Megawatt']) ==0:
        a =0
    else:
        a = sum(new_df['Megawatt']*new_df['DollarsPerMegawattHour'])/sum(new_df['Megawatt'])
    return P_max, P_min, a