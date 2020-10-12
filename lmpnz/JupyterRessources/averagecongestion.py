import pandas as pd
from lmpnz import stored_path

def get_average_congestion_charge(node="HEN0331"):
    df = pd.read_csv(stored_path.main_path + "/data/historicaLMPs/201909_Final_prices.csv")

    df = df[df["Trading_date"] == "2019-09-02"]

    actual_lmps = pd.read_csv(stored_path.main_path + "/data/historicaLMPs/Wholesale_price_trends_20200825111248.csv")
    actual_lambds = actual_lmps["Price ($/MWh)"].values

    def congestion(row):
        return row["Price"] - actual_lambds[row["Trading_period"]-1]

    df["congestion"] = df.apply(congestion, axis=1)

    congestion_node = df.groupby("Node").mean()

    test = df[df["Node"]==node]
    return congestion_node.loc[node]["congestion"], test
