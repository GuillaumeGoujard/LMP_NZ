import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from sklearn.linear_model import LinearRegression

path =  '/Users/salomeschwarz/Desktop/GitHub/LMP_NZ/drafts/SalomeCharles/'
df = pd.read_csv(path+'20190901_Offers.csv', parse_dates=['UTCSubmissionTime'])

""" Process the data """
df = df[df['ProductType'] != 'Reserve']
# drop units that bid 0 MW for all 5 bands in the last timestamp of the tp
df = df[df['Megawatt'] != 0]
# we have problems because 2 unit names for one PointOfConnection
# (eg SFD21 and SFD22) and vice versa (ROX)
# so I'm creating a column that takes 'Unit_PointOfConnection' as value
df['name'] = df['Unit'].astype(str) + '_' + df['PointOfConnection']


""" Getting one plot for one unit 
Each plot has 48 offer plots + 
one linear average approximation for the whole day 
of the unit considered """


# courbe de moyenne pour un unique géné (eg: CYD0) et les 48 trading periods correspondantes
def Offers_allTP_oneGenerator_plot(df, generator):
    fig = figure(figsize=(6, 3))
    df_ = df[df['name'] == generator]
    testdf = df_
    TP = [i for i in range(1, 49)]
    for tp in TP:
        testdf_ = testdf[testdf['TradingPeriod'] == tp]
        print('we are now doing tp ' +str(tp))
        testdf_ = testdf_[testdf_.groupby('Band').UTCSubmissionTime.transform('max') == testdf_['UTCSubmissionTime']]
        # print(testdf_)
        testdf_ = testdf_.sort_values(by='DollarsPerMegawattHour')
        testdf_['cumsum_MW'] = testdf_['Megawatt'].cumsum()

        x = testdf_['cumsum_MW']
        y = testdf_['DollarsPerMegawattHour']

        #fig = figure(figsize=(6, 3))
        plt.plot(x, y, drawstyle='steps-pre', figure=fig)
        #plt.step(x, y)
        plt.xlabel('cumsum_MW', fontsize=18)
        plt.ylabel('Dollars Per MegawattHour', fontsize=16)
        plt.title('Unit ' + str(generator) + str(tp))
        # handles, labels = ax.get_legend_handles_labels()
        # fig.legend(handles, labels, loc='upper center')
        # plt.legend(loc=2, prop={'size': 6})
        plt.plot(x, y, 'C0o', alpha=0.5)
        #plt.show()
    """Now incorporating our previous function in order to estimate our linear model"""

    df_mean = Mean_Offers_allTP_oneGenerator(df, generator)
    # my_label = 'average of all 48 \n trading periods ' + str(generator)
    # plt.plot(df_mean['cumsum_MW'], df_mean['mean'], color='black', label=my_label)

    x = df_mean['cumsum_MW']
    # X = x.values.reshape(-1,1)
    y = df_mean['mean']
    #plt.plot(x,y, color='black')
    model = np.polyfit(x, y, 1)  # returns array(a, b) such that y = a*x + b
    """credit: https://data36.com/linear-regression-in-python-numpy-polyfit/"""

    predict = np.poly1d(model)
    x_linreg = range(int(x.min()), int(x.max()) + 5)
    y_linreg = predict(x_linreg)
    # plt.scatter(x,y, color = 'black')
    my_label = 'linear approx of \n average of all 48 \n trading periods ' + str(
        generator) + 'equation is: \n y = ' + str(model[0]) + '* x + ' + str(model[1])
    #plt.plot(x_linreg, y_linreg, color='r', label=my_label)

    plt.legend()
    plt.show()
    return
# Offers_allTP_oneGenerator_plot(df, generator='BEN0_BEN2202')


# return df_mean
def Mean_Offers_allTP_oneGenerator(df, generator):
    df_mean = pd.DataFrame()  # dataframe with two columns to plot: 'mean' and 'cumsum_MW'

    df_ = df[df['name'] == generator]
    df_ = df_[['Band', 'UTCSubmissionTime', 'TradingPeriod', 'Megawatt', 'DollarsPerMegawattHour']]

    dfs = []  # list of dataframes of each tp (to merge one column later)
    lst = []  # list of unique values of MWh offers for this generator over the 48 trading periods

    for tp in range(1, 49):
        df_tp = df_[df_['TradingPeriod'] == tp]
        df_tp = df_tp[df_tp.groupby('Band').UTCSubmissionTime.transform('max') == df_tp['UTCSubmissionTime']]
        df_tp = df_tp.sort_values(by='DollarsPerMegawattHour')
        df_tp['DollarsPerMegawattHour ' + str(tp)] = df_tp['DollarsPerMegawattHour']
        df_tp['cumsum_MW'] = df_tp['Megawatt'].cumsum()

        lst.extend(pd.unique(df_tp['cumsum_MW']))
        dfs.append(df_tp)
    # print('done first loop')

    # creating the column 'cumsum_MW' on which we will merge dataframes from the list dfs
    column = list(set(lst))
    df_mean['cumsum_MW'] = column
    df_mean = df_mean.sort_values(by='cumsum_MW')
    df_mean = df_mean.reset_index(drop=True)

    # all_dfs = df_mean
    for i in range(1, 49):
        # lst.extend(pd.unique(dfs[i]['cumsum_MW']))
        data = dfs[i - 1][['cumsum_MW', 'DollarsPerMegawattHour ' + str(i)]]
        df_mean = pd.merge(df_mean, data, on='cumsum_MW', how='left')
        df_mean = df_mean.drop_duplicates()
        # print(i)
    # print('done second loop')
    df_mean = df_mean.fillna(method='backfill')
    df_mean['mean'] = df_mean.iloc[:, 1:].mean(axis=1)

    return df_mean
df_mean = Mean_Offers_allTP_oneGenerator(df, generator='BEN0_BEN2202')

# courbe noire et rouge
def Mean_Offers_allTP_oneGenerator_linreg(df, generator):
    fig = figure(figsize=(6, 3))
    plt.xlabel('cumsum_MW', fontsize=18)
    plt.ylabel('Dollars Per MegawattHour', fontsize=16)
    plt.title(str(generator))

    df_mean = Mean_Offers_allTP_oneGenerator(df, generator)

    x = df_mean['cumsum_MW']
    y = df_mean['mean']

    plt.scatter(x, y, color='blue')

    if df_mean.shape[0] >= 3:

        plt.plot(x, y, color='black')

        model = np.polyfit(x, y, 1)  # returns array(a, b) such that y = a*x + b
        """credit: https://data36.com/linear-regression-in-python-numpy-polyfit/"""

        predict = np.poly1d(model)
        x_linreg = range(int(x.min()), int(x.max()) + 8)
        y_linreg = predict(x_linreg)

        my_label = 'linear approx of \n average of all 48 \n trading periods ' + str(
            generator) + 'equation is: \n MC = ' + str(round(model[0], 5)) + '* x + ' + str(round(model[1], 5))
        plt.plot(x_linreg, y_linreg, color='r', label=my_label)

    else:
        avg = (df_mean['mean'] * df_mean['cumsum_MW']).sum()
        avg = avg / df_mean['cumsum_MW'].max()
        my_label = 'average price is : ' + str(round(avg, 5)) + '$/MW'
        plt.plot(x, y, color='black', label=my_label)

    plt.legend()
    plt.show()
    return

# array(q,a) of the linear model
def linreg(df,generator):
    generator='ROX0'
    df_mean = Mean_Offers_allTP_oneGenerator(df, generator)

    x = df_mean['cumsum_MW']
    y = df_mean['mean']

    if df_mean.shape[0] >= 3:
        model = np.polyfit(x, y, 1)  # returns array(a, b) such that y = a*x + b
        return model
    else:
        avg = (df_mean['mean'] * df_mean['cumsum_MW']).sum()
        avg = avg / df_mean['cumsum_MW'].max()
        return np.array(0, avg)
# model = linreg(df,generator)



""" Now getting our plots for 2012-12-31 :) """
Gene = pd.unique(df['name'])
for i, gene in enumerate(Gene):
    Mean_Offers_allTP_oneGenerator_linreg(df, generator=gene)
    print(gene +' done ' + str(i))



""" create my dictionaries """

""" DICT 1 """
# function that gives you a dictionary of 2 columns: unit & model
def make_regression_dict(df):
    dict = {}
    keys = pd.unique(df['name'])
    values = [linreg(df,u) for u in keys]
    for i, u in enumerate(keys):
        dict[str(u)] = values[i]
    return dict

regression_dict = make_regression_dict(df)

import json
with open('regression_dict.json', 'w') as fp:
    json.dump(regression_dict, fp)

""" DICT 2 """
""" Create an array called nodes using the ABM paper and 'pd.unique(df.name)' """
nodes = np.array(['OTA', 'ROX', 'WKM', np.nan,
          'ROX', 'ROX', 'NPL', 'NPL',
          'SFD', np.nan, np.nan, 'HLY',
          'WHI', 'WKM', 'WKM', 'HLY',
          'HLY', np.nan, 'ISL', np.nan,
          'TKU', 'WHI', 'WHI', 'WHI',
          'BPE', np.nan, np.nan, 'MAN',
          'TIW', np.nan, np.nan, np.nan,
          'HLY', 'BPE', 'HAY', 'TWZ',
          'HAY', 'HAY', np.nan, np.nan,
          np.nan, np.nan, 'WKM', np.nan,
          'WKM', 'WKM', np.nan, np.nan,
          'WKM', np.nan, 'WKM', 'WKM',
          'BPE', 'WKM', 'WKM', 'WKM',
          'NPL', np.nan, np.nan, 'STK',
          'ISL', 'BPE', 'HWB', 'ISL',
          'NPL', 'NPL', 'NPL', 'HWB',
          'HWB', 'WKM', 'IGH', 'BPE',
          'WKM', 'TWZ', 'WKM', 'STK',
          'BPE', 'WKM', 'WKM', 'HLY',])
# créer P_min
# créer P_max
# créer generation_type
# créer fuel_name
def make_generator_adjacency_matrix_dict(df):
    dict = {}
    keys = pd.unique(df['name'])
    values = nodes
    for i, u in enumerate(keys):
        dict[str(u)] = values[i]
    return dict

generator_adjacency_matrix_dict = make_regression_dict(df)

