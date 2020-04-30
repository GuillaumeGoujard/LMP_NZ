import numpy as np
import pandas as pd

import stored_path


# LMPs = pd.read_csv(stored_path.main_path + '/data/historicaLMPs/201909_Final_prices.csv')
# LMPs["Region ID"] = LMPs["Node"].apply(lambda s : s[:3])
#
# nodes = np.unique(LMPs["Region ID"])
#
# LMPs["date"] = pd.to_datetime(LMPs['Trading_date'], format="%Y-%m-%d")
# LMPs["day"] = LMPs["date"].apply(lambda s: s.day)
# LMPs = LMPs[['date', 'day', 'Trading_period', 'Region ID', 'Price']]
#
# nodes_to_index = pd.read_csv(stored_path.main_path + '/data/ABM/ABM_Nodes.csv')
# nodes_ABM = list(nodes_to_index["Node names"])


def get_vector_LMP(td, day):
    d = np.zeros(len(nodes_ABM))
    for i, n in enumerate(nodes_ABM):
        if n in nodes:
            d[i] = np.mean(LMPs[(LMPs["Region ID"] == n) & (LMPs["Trading_period"] == td) & (LMPs["day"] == day)]["Price"])
    return d

path = stored_path.main_path + '/drafts/SalomeCharles/Data/'

def create_data_Sept():

    df1 = pd.read_csv(path + '20190901_Offers.csv', parse_dates=['UTCSubmissionTime'])
    df2 = pd.read_csv(path + '20190902_Offers.csv', parse_dates=['UTCSubmissionTime'])
    df3 = pd.read_csv(path + '20190903_Offers.csv', parse_dates=['UTCSubmissionTime'])
    df4 = pd.read_csv(path + '20190904_Offers.csv', parse_dates=['UTCSubmissionTime'])
    df5 = pd.read_csv(path + '20190905_Offers.csv', parse_dates=['UTCSubmissionTime'])
    df6 = pd.read_csv(path + '20190906_Offers.csv', parse_dates=['UTCSubmissionTime'])
    df7 = pd.read_csv(path + '20190907_Offers.csv', parse_dates=['UTCSubmissionTime'])

    dfs = []
    dfs_ = [df1, df2, df3, df4, df5, df6, df7]
    for i, df in enumerate(dfs_):
        df = df[df['ProductType'] != 'Reserve']

        # drop units that bid 0 MW for all 5 bands in the last timestamp of the tp
        # df = df[df['Megawatt'] != 0]

        # we have problems because 2 unit names for one PointOfConnection (eg SFD21 and SFD22) and vice versa (ROX)
        # so I'm creating a column that takes 'Unit_PointOfConnection' as value
        df['name'] = df['Unit'].astype(str) + '_' + df['PointOfConnection']

        df["date"] = pd.to_datetime(df["TradingDate"], format="%Y-%m-%d")
        df["day"] = df["date"].swifter.apply(lambda s: s.day)
        # new value for trading period
        # df['TradingPeriod'] = df['TradingPeriod'].astype(float) * (i + 1)

        dfs.append(df)

    df = pd.concat(dfs)

    return df


df = create_data_Sept()

def get_P_min_a(generator, day,  tp, type):
    df_ = df[(df['name'] == generator)&(df['TradingPeriod'] == tp)&(df['day'] == day)].copy()
    # df_ = df_[df_.groupby('Band').UTCSubmissionTime.transform('max') == df_['UTCSubmissionTime']]
    df_ = df_[df_["IsLatest"]=="Y"]
    df_ = df_.sort_values(by='DollarsPerMegawattHour')
    df_['cumsum_MW'] = df_['Megawatt'].cumsum()

    min_power = df_['cumsum_MW'][df_.DollarsPerMegawattHour <= 5]
    # if type == "Hydro":
    P_min = min_power.iloc[-1] if len(min_power) >0  else 0
    cap = 5
    # else:
    #     P_min = 0
    #     cap = 5
    P_max = df_['cumsum_MW'].iloc[-1] if len(df_) >0  else 0
    new_df = df_[df_.DollarsPerMegawattHour >= cap].copy()
    if len(new_df) == 0 or sum(new_df['cumsum_MW']) == 0 or sum(new_df['Megawatt']) ==0:
        a =0
    else:
        a = sum(new_df['Megawatt']*new_df['DollarsPerMegawattHour'])/sum(new_df['Megawatt'])
    return P_max, P_min, a



# import json
# file_path = stored_path.main_path + '/data/generators/generator_adjacency_matrix_dict.json'
# with open(file_path) as f:
#     data = json.loads(f.read())
#
# generator = "KIN0_KIN0112"
# day =1
# tp = 24
# type = "Hydro"

#
# tp = 15
# day = 1

# tp =1
# day = 2
# generator = 'GLN0_GLN0332'
# df[(df['TradingPeriod'] == tp)&(df['day'] == day)].copy()
# P_max, P_min, a = get_P_min_a('GLN0_GLN0332', 1, 2)
# Pmins = []
# as_ = []
# for name_generator in data.keys():
#     L_ = data[name_generator]
#     P_min, a = get_P_min_a(name_generator, 1, 1)
#     Pmins.append(P_min)
#     as_.append(a)



