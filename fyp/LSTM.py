import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt


def main():
    try:
        f = open("data/traffic_model.hd5")
        model = load_model("data/traffic_model.hd5")
        print("Loading existing model...")
    except IOError:
        print("No existing model found. Building new model...")

        model = build_traffic_model()
    finally:
        f.close()
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
    rmse=np.sqrt(np.mean(((predictions- y_test)**2)))
    print(rmse)

    plot_traffic(df, predictions)


def traffic_preproc():
    traffic_data = pd.read_csv('data/Kirklees_csv.csv',
                               usecols=['local_authority_name', 'count_date', 'all_motor_vehicles'],
                               parse_dates=['count_date'])
    return traffic_data


def plot_traffic(df, predictions):
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
    print(df.values)


def build_traffic_model():
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
    model.save("data/traffic_model.hd5")
    return model


# PREP DATA
df = traffic_preproc()

# Separate vehicle column
data = df.filter(['all_motor_vehicles'])

# Convert the df to a numpy array
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


if __name__ == '__main__':
    main()
