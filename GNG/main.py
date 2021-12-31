import tensorflow as tf
import pandas as pd
import os

from GrowingNeuralGas import GrowingNeuralGas
from utils import train_and_test

def test():
    tf.random.set_seed(23)
    X = tf.concat([tf.random.normal([500, 2], 0.0, 0.25, dtype=tf.float32, seed=1) + tf.constant([0.0, 0.0]),
                   tf.random.normal([500, 2], 0.0, 0.25, dtype=tf.float32, seed=1) + tf.constant([1.0, 0.0]),
                   tf.random.normal([500, 2], 0.0, 0.25, dtype=tf.float32, seed=1) + tf.constant([1.0, 1.0])], 0)

    growingNeuralGas = GrowingNeuralGas()
    growingNeuralGas.fit(X, 5)

    pass

def test_vinos():

    data_path = os.getcwd() + "/datasets"
    file = "/vinos_pca"
    path = os.getcwd() + "/data" + file

    data = pd.read_csv(data_path + file + ".csv")

    training_batch, test_batch = train_and_test(data, 0.8)

    epsilon_a = .1
    epsilon_n = .01
    a_max = 5
    eta = 20
    epochs = 10
    alpha = .15
    delta = .15

    endpath = "-A_" + str(epsilon_a) + "-N_" + str(epsilon_n) + "-Max_" + str(a_max) + "-eta_" + str(eta) \
              + "-epochs_" + str(epochs) + "-alpha_" + str(alpha) + "-delta_" + str(delta)

    tensor = tf.convert_to_tensor(training_batch, dtype=tf.float32)

    if not (os.path.exists(path + endpath)):
        os.mkdir(path + endpath)

    growingNeuralGas = GrowingNeuralGas(epsilon_a=epsilon_a, epsilon_n=epsilon_n, a_max=a_max, eta=eta, alpha=alpha,
                                        delta=delta)
    growingNeuralGas.fit(tensor, epochs, 100, path + endpath)

    growingNeuralGas.saver(os.getcwd() + "/saves" + file + endpath)
    # growingNeuralGas.loader(os.getcwd() + "/saves"+ file + endpath)
    pass

test_vinos()


