import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
path =  '/Users/salomeschwarz/Desktop/GitHub/LMP_NZ/drafts/SalomeCharles/'

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
        df = df[df['Megawatt'] != 0]

        # we have problems because 2 unit names for one PointOfConnection (eg SFD21 and SFD22) and vice versa (ROX)
        # so I'm creating a column that takes 'Unit_PointOfConnection' as value
        df['name'] = df['Unit'].astype(str) + '_' + df['PointOfConnection']

        # new value for trading period
        df['TradingPeriod'] = df['TradingPeriod'].astype(float) * (i + 1)

        dfs.append(df)

    df = pd.concat(dfs)
    df.to_csv(path + '20190901-07_Offers.csv', index=False)

    return

df = pd.read_csv(path + '20190901-07_Offers.csv', parse_dates=['UTCSubmissionTime'])

""" Getting one plot for one unit 
Each plot has 48 offer plots + 
one linear average approximation for the whole day 
of the unit considered """

# courbe de moyenne pour un unique géné (eg: CYD0) et les 48 trading periods correspondantes
def Offers_allTP_oneGenerator_plot(df, generator):
    fig = figure(figsize=(8, 5))
    df_ = df[df['name'] == generator]
    testdf = df_
    TP = [i for i in range(1, 337)]
    for tp in TP:
        testdf_ = testdf[testdf['TradingPeriod'] == tp]
        print('we are now doing tp ' +str(tp))
        testdf_ = testdf_[testdf_.groupby('Band').UTCSubmissionTime.transform('max') == testdf_['UTCSubmissionTime']]
        #print(testdf_[['DollarsPerMegawattHour', 'Megawatt']])
        testdf_ = testdf_.sort_values(by='DollarsPerMegawattHour')
        testdf_['cumsum_MW'] = testdf_['Megawatt'].cumsum()

        x = testdf_['cumsum_MW']
        y = testdf_['DollarsPerMegawattHour']

        #fig = figure(figsize=(6, 3))
        plt.plot(x, y, drawstyle='steps-pre', figure=fig)
        #plt.step(x, y)
        plt.xlabel('cumsum_MW', fontsize=18)
        plt.ylabel('Dollars Per MegawattHour', fontsize=16)
        plt.title('Unit ' + str(generator))
        # handles, labels = ax.get_legend_handles_labels()
        # fig.legend(handles, labels, loc='upper center')
        # plt.legend(loc=2, prop={'size': 6})
        plt.plot(x, y, 'C0o', alpha=0.5)
        #plt.show()
    plt.legend()
    plt.show()
    return
# Offers_allTP_oneGenerator_plot(df, generator='BEN0_BEN2202')
# Offers_allTP_oneGenerator_plot(df, generator='ROX0_ROX2201')
# Offers_allTP_oneGenerator_plot(df, generator='ROX0_ROX1101')

# return df_mean
def Mean_Offers_allTP_oneGenerator(df, generator):
    df_mean = pd.DataFrame()  # dataframe with two columns to plot: 'mean' and 'cumsum_MW'

    df_ = df[df['name'] == generator]
    df_ = df_[['Band', 'UTCSubmissionTime', 'TradingPeriod', 'Megawatt', 'DollarsPerMegawattHour']]

    dfs = []  # list of dataframes of each tp (to merge one column later)
    lst = []  # list of unique values of MWh offers for this generator over the 48 trading periods

    for tp in range(1, 337):
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
    for i in range(1, 337):
        # lst.extend(pd.unique(dfs[i]['cumsum_MW']))
        data = dfs[i - 1][['cumsum_MW', 'DollarsPerMegawattHour ' + str(i)]]
        df_mean = pd.merge(df_mean, data, on='cumsum_MW', how='left')
        df_mean = df_mean.drop_duplicates()
        # print(i)
    # print('done second loop')
    df_mean = df_mean.fillna(method='backfill')
    df_mean['mean'] = df_mean.iloc[:, 1:].mean(axis=1)

    df_mean = df_mean[['mean', 'cumsum_MW']]
    return df_mean
# df_mean = Mean_Offers_allTP_oneGenerator(df, generator='BEN0_BEN2202')
# df_mean = Mean_Offers_allTP_oneGenerator(df, generator='ROX0_ROX1101')


# courbe noire (scatter) et rouge pour chaque générateur
def Mean_Offers_allTP_oneGenerator_linreg_plot_daft(df, generator):
    fig = figure(figsize=(8, 5))
    plt.xlabel('cumsum_MW', fontsize=18)
    plt.ylabel('Dollars Per MegawattHour', fontsize=16)
    plt.title(str(generator))

    df_mean = Mean_Offers_allTP_oneGenerator(df, generator)

    #df_mean =  df_mean.dropna(how='any')

    # scatter plot
    x = df_mean['cumsum_MW']
    y = df_mean['mean']
    #print(y)
    plt.scatter(x, y, color='blue', figure=fig)

    # linear regression plot
    x_1 = df_mean['cumsum_MW'].loc[0]
    x_x_1 = (df_mean['cumsum_MW']-x_1).to_frame()
    Y = df_mean['mean'].to_frame()
    #b = df_mean['mean'].min().astype(float)
    b = df_mean['mean'].loc[0]
    #print(b)

    # Ols model
    pred_ols = sm.OLS(Y -b, x_x_1).fit()
    model_ols = pred.params
    print(model_ols)

    # Wls model 1
    # w = [20000] + [1]*(len(x)-1)
    # pred_wls_1 = sm.WLS(Y -b, x_, weights=w).fit()
    # model_wls_1 = pred_wls_1.params
    #
    # # Wls model 2
    # df_mean['intercept'] = 1
    # X = df_mean.drop('mean', axis=1)
    # pred_wls_2 = sm.WLS(Y, X, weights=w).fit() # weights à mon modèle, mais pas de
    # # contraintes sur b
    # model_wls_2 = pred_wls_2.params
    # print(model_wls_2)
    #
    # my_label_1 = 'wls_model, intercept constrained, weights = ' +str(w[0])
    # my_label_2 = 'wls_model, weights = ' + str(w[0])
    plt.plot(np.array(df_mean['cumsum_MW']), pred_ols.predict(), label='ols model', figure=fig, color='r')
    # plt.plot(np.array(df_mean['cumsum_MW']), pred_wls_1.predict()+b , label=my_label_1, figure=fig, color='green')
    # plt.plot(np.array(df_mean['cumsum_MW']), pred_wls_2.predict() , label=my_label_2, figure=fig, color='black')


    plt.legend()
    plt.show()

    #plt.plot(x, y, color='black', figure=fig)

        #df_mean['intercept'] = 1
        #X = df_mean.drop('mean', axis=1)
        #print(b)
        #model1 = sm.OLS(Y - b, X).fit().params
        #pred1 = sm.OLS(Y - b, X).fit()
        #print(list(round(model, 3))) # c'est le paramètre a

        #plt.plot(np.array(df_mean['cumsum_MW']), pred1.predict(), label='mod1', figure=fig)
        #plt.plot(np.array(df_mean['cumsum_MW']), pred2.predict(), label='mod2', figure=fig)
        #my_label = 'linear approx of \n average bidding price \n on all trading periods \n' + str('generator') + 'equation is: \n MC = ' + str(round(model[0], 5)) + '* x + '
        #plt.plot(x_linreg, b + model[0] * x_linreg, color='r', label = my_label, figure=fig)
        #plt.plot(x_linreg, model[0] * x_linreg, color='blue', label=my_label)
        # model = np.polyfit(x, y, 1)  # returns array(a, b) such that y = a*x + b
        # """credit: https://data36.com/linear-regression-in-python-numpy-polyfit/"""
        #
        # predict = np.poly1d(model)
        # x_linreg = range(int(x.min()), int(x.max()) + 8)
        # y_linreg = predict(x_linreg)
        # plt.plot(x_linreg, y_linreg, color='r', label=my_label)
    return

def Mean_Offers_allTP_oneGenerator_linreg_plot(df, generator):
    fig = figure(figsize=(8, 5))
    plt.xlabel('cumsum_MW', fontsize=18)
    plt.ylabel('Dollars Per MegawattHour', fontsize=16)
    plt.title(str(generator))

    df_mean = Mean_Offers_allTP_oneGenerator(df, generator)

    # scatter plot
    x = df_mean['cumsum_MW']
    y = df_mean['mean']
    plt.scatter(x, y, color='darkturquoise', figure=fig)

    # linear regression data processing
    ## process x
    x_1 = df_mean['cumsum_MW'].loc[0]
    x_x_1 = (df_mean['cumsum_MW']-x_1).to_frame()
    ## process y
    Y = df_mean['mean'].to_frame()
    b = df_mean['mean'].loc[0]

    # Ols model
    pred_ols = sm.OLS(Y - b, x_x_1).fit() # you force your first point to be in the ols line
    model_ols = pred_ols.params
    a = round(model_ols[0], 3)
    b = round(b,2)

    # make the plot
    my_label = 'ols model \n price = '+ str(a)+ 'x + ' +str(b)
    plt.plot(np.array(df_mean['cumsum_MW']), pred_ols.predict()+b, label=my_label, figure=fig, color='orchid', linewidth=2)
    plt.legend()
    plt.grid()
    plt.show()
    return

# Mean_Offers_allTP_oneGenerator_linreg_plot(df, generator='TKU0_TKU2201')
# Mean_Offers_allTP_oneGenerator_linreg_plot(df, generator='TUI0_TUI1101')

# [a,b] from the linear model
def linreg(df,generator):

    df_mean = Mean_Offers_allTP_oneGenerator(df, generator)

    # linear regression data processing
    ## process x
    x_1 = df_mean['cumsum_MW'].loc[0]
    x_x_1 = (df_mean['cumsum_MW'] - x_1).to_frame()
    ## process y
    Y = df_mean['mean'].to_frame()
    b = df_mean['mean'].loc[0]

    # Ols model
    pred_ols = sm.OLS(Y - b, x_x_1).fit()  # you force your first point to be in the ols line
    model_ols = pred_ols.params

    # list(model_ols) = [a]
    # b is fixed
    a = round(model_ols[0], 3)
    b = round(b, 2)
    return [a, b]

# linreg(df,generator) = [a,b]

linreg(df,generator='TKU0_TKU2201')

#######################################
""" create my dictionary """
#######################################


""" DICT 1 """
# function that gives you a dictionary of 2 columns: unit & model

""" Create an array called nodes using the ABM paper and 'pd.unique(df.name)' """
nodes = ['OTA', 'ROX', 'WKM', np.nan,
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
          'BPE', 'WKM', 'WKM', 'HLY',]
""" Get Pmax and type for my dictionay"""

## process data of generating plants
df_plant = pd.read_csv(path + 'generating_plant.csv', sep=';')[['Fuel_Name', 'Installed_Capacity', 'Node_Name']]
df_plant = df_plant.append(df_plant.loc[23])
df_plant.reset_index(drop=True, inplace=True)
df_plant.loc[23,'Node_Name'] = 'WWD1102'
df_plant.loc[225,'Node_Name'] = 'WWD1103'

## create new dataframe
#cols will be ['Node_Name' (PointOfConnection), 'Fuel_Name', 'Installed_Capacity' ]

df_Pmax_and_type = pd.DataFrame()
df_Pmax_and_type['Node_Name'] = df.PointOfConnection
df_Pmax_and_type['Unit'] = df.Unit
df_Pmax_and_type['name'] = df_Pmax_and_type['Unit'].astype(str) + '_' + df_Pmax_and_type['Node_Name'].astype(str)
df_Pmax_and_type = df_Pmax_and_type.drop_duplicates()
df_Pmax_and_type.reset_index(drop=True, inplace=True)
df_Pmax_and_type = pd.merge(df_Pmax_and_type, df_plant, on='Node_Name', how = 'left')
df_Pmax_and_type['Installed_Capacity'] = pd.to_numeric(df_Pmax_and_type.Installed_Capacity.str.replace(',','.'), errors='coerce')

def make_generator_adjacency_matrix_dict(df):
    """
    output key=name of unit , value = [nodeABM, [a,b], Pmin, Pmax, fuel_name]
    :param df:
    :return:
    """
    dict = {}
    keys = pd.unique(df['name'])

    nodeABM = nodes
    ab = [linreg(df,u) for u in keys]
    Pmin = [0]*len(keys)
    ## getting Pmax
    Pmax = list(df_Pmax_and_type.Installed_Capacity.astype(float))
    ## getting fuel_name
    Fuel_name = list(df_Pmax_and_type.Fuel_Name)

    for i, u in enumerate(keys):
        dict[str(u)] = [nodeABM[i], ab[i], Pmin[i], Pmax[i], Fuel_name[i]]
    return dict

generator_adjacency_matrix_dict = make_generator_adjacency_matrix_dict(df)



test_dict = {"a":1, "b":[2,4,6] }


import json
with open(path+'generator_adjacency_matrix_dict.json', 'w') as fp:
    json.dump(generator_adjacency_matrix_dict, fp)



