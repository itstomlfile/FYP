import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt


def main():
    # PREP DATA
    traffic_df = traffic_preproc()

    # Separate vehicle column
    traffic_data = traffic_df.filter(['all_motor_vehicles'])

    # Convert the traffic_df to a numpy array
    traffic_dataset = traffic_data.values

    # Get the number of rows to train the model on (80%)
    traffic_training_data_len = math.ceil(len(traffic_dataset) * .8)

    # Scale the traffic_data
    scaler = MinMaxScaler(feature_range=(0, 1))
    traffic_scaled_data = scaler.fit_transform(traffic_dataset)  # Scaled values between 0 and 1

    # Create scaled training traffic_data set
    traffic_train_data = traffic_scaled_data[0:traffic_training_data_len, :]

    # Split the traffic_data
    traffic_x_train = []
    traffic_y_train = []

    for i in range(60, len(traffic_train_data)):
        traffic_x_train.append(traffic_train_data[i - 60:i, 0])
        traffic_y_train.append(traffic_train_data[i, 0])

    # Convert the traffic_x_train and traffic_y_train to numpy arrays
    traffic_x_train, traffic_y_train = np.array(traffic_x_train), np.array(traffic_y_train)

    # Reshape the traffic_data
    traffic_x_train = np.reshape(traffic_x_train, (traffic_x_train.shape[0], traffic_x_train.shape[1], 1))

    try:
        f_traffic = open("data/traffic_model.hd5")
        model = load_model("data/traffic_model.hd5")
        print("Loading existing model...")
    except IOError:
        print("No existing model found. Building new model...")

        model = build_model(traffic_x_train, traffic_y_train, "data/traffic_model.hd5")
    finally:
        f_traffic.close()
    # Create the testing traffic_data set
    traffic_test_data = traffic_scaled_data[traffic_training_data_len - 60:, :]
    # Create the traffic_data sets traffic_x_test and traffic_y_test
    traffic_x_test = []
    traffic_y_test = traffic_dataset[traffic_training_data_len:, :]

    for i in range(60, len(traffic_test_data)):
        traffic_x_test.append(traffic_test_data[i - 60:i, 0])

    # Convert the traffic_data to a numpy array
    traffic_x_test = np.array(traffic_x_test)

    # Reshape the traffic_data
    traffic_x_test = np.reshape(traffic_x_test, (traffic_x_test.shape[0], traffic_x_test.shape[1], 1))
    traffic_predictions = model.predict(traffic_x_test)
    traffic_predictions = scaler.inverse_transform(traffic_predictions)

    # Get the root mean squared error (RMSE)
    rmse = np.sqrt(np.mean(((traffic_predictions - traffic_y_test)**2)))
    print(rmse)

    plot_traffic(traffic_predictions, traffic_data, traffic_training_data_len)


def traffic_preproc():
    traffic_data = pd.read_csv('data/Kirklees_csv.csv',
                               usecols=['local_authority_name', 'count_date', 'all_motor_vehicles'],
                               parse_dates=['count_date'])
    return traffic_data


def plot_traffic(predictions, data, training_data_len):
    # Plot the data
    train = data[:training_data_len]
    valid = data[training_data_len:]
    valid['Predictions'] = predictions
    plt.figure(figsize=(16,8), dpi=128)
    plt.title('Traffic 2001-2005')
    plt.xlabel('Year', fontsize=18)
    plt.ylabel('Number of Vehicles', fontsize=18)
    plt.plot(train['all_motor_vehicles'])
    plt.plot(valid[['all_motor_vehicles', 'Predictions']])
    plt.legend(['Training Data', 'Actual', 'Predictions'])
    # TODO: Get the x-axis to use the timestamps
    plt.show()
    plt.savefig('graphs/traffic_LSTM_predictions.png')


def build_model(x_train, y_train, name):
    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(x_train, y_train, batch_size=1, epochs=5)
    model.save(name)
    return model


if __name__ == '__main__':
    main()
