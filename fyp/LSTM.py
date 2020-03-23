import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt


def lstm_prep_and_predict(df, model_path, dependent):
    # Separate vehicle column
    data = df.filter([dependent])

    # Convert the df to a numpy array
    # df['Start time'] = pd.to_datetime(df['Start time'])
    dataset = data.values

    # Get the number of rows to train the model on (80%)
    training_data_len = math.ceil(len(dataset) * .8)

    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)  # Scaled values between 0 and 1

    # Create scaled training data set
    train_data = scaled_data[0:training_data_len, :]

    # Split the data
    x_train = []
    y_train = []

    for i in range(60, len(train_data)):
        x_train.append(train_data[i - 60:i, 0])
        y_train.append(train_data[i, 0])

    # Convert the x_train and y_train to numpy arrays
    x_train, y_train = np.array(x_train), np.array(y_train)

    # Reshape the data
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    try:
        f = open(model_path)
        model = load_model(model_path)
        print("Loading existing model...")
        f.close()
    except FileNotFoundError:
        print("No existing model found. Building new model...")
        model = build_model(x_train, y_train, model_path)
    # Create the testing data set
    test_data = scaled_data[training_data_len - 60:, :]
    # Create the data sets x_test and y_test
    x_test = []
    y_test = dataset[training_data_len:, :]

    for i in range(60, len(test_data)):
        x_test.append(test_data[i - 60:i, 0])

    # Convert the data to a numpy array
    x_test = np.array(x_test)

    # Reshape the data
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)

    # Get the root mean squared error (RMSE)
    rmse=np.sqrt(np.mean(((predictions - y_test)**2)))
    print(rmse)

    return df, predictions, training_data_len


def traffic_preproc():
    traffic_data = pd.read_csv('data/Kirklees_csv.csv',
                               usecols=['local_authority_name', 'count_date', 'all_motor_vehicles'],
                               parse_dates=['count_date'])
    return traffic_data


def emissions_preproc():
    emissions_data = pd.read_csv('data/emissions_csv.csv',
                               usecols=['Start time', 'NO2'])
    # Remove empty fields to stop skewing learning
    emissions_data['NO2'].replace('',np.nan, inplace=True)
    emissions_data.dropna(subset=['NO2'], inplace=True)
    return emissions_data


def plot_graph(predictions, data, training_data_len, title, x_label, y_label, dependant, fig):
    # Plot the data
    data.index = pd.to_datetime(data.index)
    train = data[:training_data_len]
    valid = data[training_data_len:]
    valid['Predictions'] = predictions
    plt.figure(figsize=(16,8), dpi=256)
    plt.title(title)
    plt.xlabel(x_label, fontsize=18)
    plt.ylabel(y_label, fontsize=18)
    plt.plot(train[dependant])
    plt.plot(valid[[dependant, 'Predictions']])

    plt.legend(['Training Data', 'Actual', 'Predictions'])
    # TODO: Get the x-axis to use the timestamps
    plt.show()
    plt.savefig(fig)


def build_model(x_train, y_train, name): 
    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))

    # Compile the model
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

    # Train the model
    model.fit(x_train, y_train, batch_size=128, epochs=10)
    model.save(name)
    return model


if __name__ == '__main__':
    # PREP DATA
    #traffic_df = traffic_preproc()
    #df, predictions, training_data_len = lstm_prep_and_predict(traffic_df, "data/traffic_model.hd5", 'all_motor_vehicles')
    #plot_graph(predictions, df, training_data_len, 'Traffic 2001-2005', 'Date', 'NO2', 'all_motor_vehicles')

    emissions_df = emissions_preproc()

    df, predictions, training_data_len = lstm_prep_and_predict(emissions_df, "data/emissions_model.hd5", 'NO2')
    plot_graph(predictions, df, training_data_len, 'Emissions 2007-2012 ', 'Date', 'NO2', 'NO2', 'graphs/emissions_LSTM_predictions.png')

