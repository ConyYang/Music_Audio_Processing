import numpy as np
from random import random
import tensorflow as tf
from sklearn.model_selection import train_test_split

# input array([[0.1, 0.2], [0.2, 0.2]])
# output array([[0.3],[0.4]])


def generate_dataset(number_of_samples, test_size):
    x = np.array([[random()/2 for _ in range(2)] for _ in range(number_of_samples)])
    y = np.array([[i[0] + i[1]] for i in x])
    # Split the dataset
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)
    # print(x)
    # print(y)
    print("Data Generated with X_train{}, y_train{}, "
          "X_test{}, y_test{}".format(x_train.shape, y_train.shape, x_test.shape, y_test.shape))
    return x_train, x_test, y_train, y_test

if __name__ == "__main__":
    x_train, x_test, y_train, y_test = generate_dataset(5000, 0.2)
    # build model 2->5->1
    model = tf.keras.Sequential(
        [tf.keras.layers.Dense(5, input_dim=2, activation="sigmoid"),
         tf.keras.layers.Dense(1, activation="sigmoid")])
    # compile the model
    optimizer = tf.keras.optimizers.SGD(lr=0.1)
    model.compile(optimizer=optimizer, loss="MSE")

    # train_model
    model.fit(x=x_train, y=y_train, epochs=100)

    # evaluate the model
    print("\nModel Evaluation")
    model.evaluate(x_test, y_test, verbose=True)

    # Make prediction
    data = np.array([[0.1, 0.2],[0.2,0.2]])
    predictions = model.predict(data)

    print("\nMake Predictions")
    print(predictions)
