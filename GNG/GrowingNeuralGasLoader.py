import numpy as np
import tensorflow as tf
import os
import json

from Graph import Graph

class GrowingNeuralGasLoader(object):
    def loadGNG(self, dirpath):
        if not (os.path.exists(dirpath)):
            print("Path not found")
        else:
            A = np.loadtxt(dirpath + "/A_save.csv", delimiter=",")
            error_ = np.loadtxt(dirpath + "/A_save.csv", delimiter=",")
            A = tf.Variable(A, dtype="float32")
            error_ = tf.Variable(error_, dtype="float32")

            n = json.loads(open(dirpath + "/N.json", "r").read())
            N = [Graph(i[1], [tf.constant(j, dtype=tf.int64) for j in i[2]], i[0]) for i in n]

            file = open(dirpath + "/variables.json", "r").read()
            pack = json.loads(file)

            return A, N, error_, pack