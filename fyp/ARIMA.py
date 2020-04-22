from statsmodels.tsa.ar_model import AR
from sklearn.metrics import mean_squared_error
import pandas as pd
from pandas import datetime
import matplotlib.pyplot as plt


def traffic_preproc():
    traffic_data = pd.read_csv('data/Kirklees_Traffic_csv.csv',
                               usecols=['count_date', 'all_motor_vehicles'],
                               index_col=['count_date'], parse_dates=['count_date'], date_parser=date_parser)
    return traffic_data


def date_parser(x):
    return pd.datetime.strptime(x, '%d/%m/%Y')


if __name__ == '__main__':
    # PREP DATA
    traffic = traffic_preproc()
    traffic.sort_values(by='count_date')
    print(traffic.head())
    plt.plot(traffic[:48])
    print(traffic[:48])
    plt.show()