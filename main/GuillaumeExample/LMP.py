import numpy as np
import pandas as pd
import swifter
import stored_path
import datetime as datetime

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
    P_min = min_power.iloc[-1] if len(min_power) >0  else 0
    P_max = df_['cumsum_MW'].iloc[-1] if len(df_) >0  else 0
    new_df = df_[df_.DollarsPerMegawattHour > cap].copy()
    if len(new_df) == 0 or sum(new_df['cumsum_MW']) == 0 or sum(new_df['Megawatt']) ==0:
        a =0
    else:
        a = sum(new_df['Megawatt']*new_df['DollarsPerMegawattHour'])/sum(new_df['Megawatt'])
    return P_max, P_min, a



