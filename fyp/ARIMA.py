import itertools
import warnings
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.preprocessing import MinMaxScaler


def prep_and_predict(df, model_path, dependent):
    data = df.filter([dependent])

    # Convert the df to a numpy array
    dataset = data.values

    # Get the number of rows to train the model on (80%)
    training_data_len = math.ceil(len(dataset) * .8)

    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)  # Scaled values between 0 and 1
    print(scaled_data.shape)
    # build the model
    model = sm.tsa.statespace.SARIMAX(scaled_data,
                                    order=(1, 1, 1),
                                    seasonal_order=(1, 0, 1, 12),
                                    enforce_stationarity=False,
                                    enforce_invertibility=False)

    results = model.fit()

    # Predicting the last year
    predictions = results.get_prediction(start=training_data_len, dynamic=False)

    # building a dataframe of predicted results
    df_predicted = predictions.predicted_mean

    # Reshaping data
    df_predicted = df_predicted.reshape(-1, 1)
    df_predicted = scaler.inverse_transform(df_predicted)


    # Plot the data
    train = df[:training_data_len]
    valid = df[training_data_len:]
    valid['Predictions'] = df_predicted
    plt.figure(figsize=(16, 8), dpi=256)
    plt.title(" ")
    plt.xlabel("Year", fontsize=18)
    plt.ylabel("Count", fontsize=18)
    plt.plot(train[dependent])
    plt.plot(valid[[dependent, 'Predictions']])
    plt.legend(['Training Data', 'Actual', 'Predictions'])
    plt.show()


def traffic_preproc():
    traffic_data = pd.read_csv('data/Kirklees_Traffic_csv.csv',
                               usecols=['count_date', 'all_vehicles'],
                               index_col=['count_date'], parse_dates=['count_date'], date_parser=date_parser)
    return traffic_data


def date_parser(x):
    return pd.datetime.strptime(x, '%d/%m/%Y')


def FindModelParams(df):
    warnings.filterwarnings("ignore")  # specify to ignore warning messages
    # The term bfill means that we use the value before filling in missing values - not used here but can be will realworld data
    df = df.fillna(df.bfill())
    dataset = df['all_vehicles']

    # Most commonly used time-series forecasting is ARIMA (Autoregressive Integrated Moving Average)
    # Seasonal ARIMA is denoted as ARIMA(p,d,q)(P,D,Q)s accounting for seasonality, trend, and noise in data.

    # p is the auto-regressive part of the model. It allows us to incorporate the effect of past values into our model.

    # d is the integrated part of the model. This includes terms in the model that incorporate the amount of differencing to apply to the time series.

    # q is the moving average part of the model. This allows us to set the error of our model as a linear combination of the error values observed at previous time points in the past.

    # (P, D, Q) follow the same definition but are applied to the seasonal component of the time series.

    # s is the periodicity of the time series (4 for quarterly periods, 12 for yearly periods, etc.).

    # Create parameter combinations for Seasonal ARIMA
    # Find the values of ARIMA(p,d,q)(P,D,Q)s that optimise a metric of interest.
    # Automate the process of identifying the optimal set of parameters for the seasonal ARIMA time series model.
    p = d = q = range(0, 3)
    pdq = list(itertools.product(p, d, q))
    seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]

    # Parameter selection for ARIMA Time Series Model.
    # Use a “grid search” to find the optimal set of parameters that yields the best performance for our model.
    # This takes a long time!
    AIC_val = []
    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                mod = sm.tsa.statespace.SARIMAX(dataset,
                                                order=param,
                                                seasonal_order=param_seasonal,
                                                enforce_stationarity=False,
                                                enforce_invertibility=False)

                results = mod.fit(disp=0)

                print('SARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
                AIC_val.append(results.aic)
            except:
                continue
    AIC_val.sort()
    print(AIC_val[0])


if __name__ == '__main__':
    # PREP DATA
    traffic = traffic_preproc()
    FindModelParams(traffic)
    #prep_and_predict(traffic, "", 'all_vehicles')