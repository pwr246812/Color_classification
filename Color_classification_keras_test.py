import csv
from tensorflow import keras
import numpy as np
from tensorflow.keras import layers


def get_data(file_name):
    data = []

    with open(file_name) as file:
        file_readed = csv.reader(file, delimiter=',')
        line = 0
        for row in file_readed:
            data.append([[float(row[0]), float(row[1]), float(row[2])], int(row[3])])
            line += 1

    return data


def split_data(data):
    x, y = [], []

    for i in range(len(data)):
        temp = [0, 0, 0, 0, 0]
        x.append(data[i][0])
        temp[data[i][1]] = 1
        y.append(np.array(temp))

    return np.array(x), np.array(y)


def normalize(data, reverse, maximum, minimum, low, high):
    if reverse:
        data = (data - low)/(high - low) * (maximum - minimum) + minimum
    else:
        data = (high - low)*(data - minimum)/(maximum - minimum) + low

    return data


train_DATA = get_data('rgb.csv')
x_train, y_train = split_data(train_DATA)
max = int(x_train.max())
min = int(x_train.min())
x_train = normalize(x_train, reverse=False, maximum=max, minimum=min, low=-1, high=1)


model = keras.Sequential()
model.add(layers.Dense(6, activation='tanh', name='layer1'))
model.add(layers.Dense(5))
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, epochs=5000, verbose=1)

print("\nTEST: \n")
errors = 0
for i in range(len(x_train)):
    prediction = model.predict(np.array([x_train[i]]))
    print(f"{i+1}. Prediction: {np.argmax(prediction[0])}   Expected: {np.argmax(y_train[i])}")
    if np.argmax(prediction[0]) != np.argmax(y_train[i]):
        errors += 1

print("\nNumber of mistakes:", errors)
if not errors:
    print("Success!")
