import functools
import os

import numpy as np
import tensorflow as tf

from Graph import Graph
from GrowingNeuralGasPlotter import GrowingNeuralGasPlotter
from GrowingNeuralGasSaver import GrowingNeuralGasSaver
from GrowingNeuralGasLoader import GrowingNeuralGasLoader

class GrowingNeuralGas(object):

    def __init__(self, epsilon_a=.1, epsilon_n=.05, a_max=10, eta=5, alpha=.1, delta=.1, maxNumberUnits=1000):
        self.A = None
        self.N = []
        self.error_ = None
        self.epsilon_a = epsilon_a
        self.epsilon_n = epsilon_n
        self.a_max = a_max
        self.eta = eta
        self.alpha = alpha
        self.delta = delta
        self.maxNumberUnits = maxNumberUnits

    def incrementAgeNeighborhood(self, indexNearestUnit):
        self.N[indexNearestUnit].incrementAgeNeighborhood(1.0)
        for indexNeighbour in self.N[indexNearestUnit].neighborhood:
            self.N[indexNeighbour].incrementAgeNeighbour(indexNearestUnit, 1.0)

    def findNearestUnit(self, xi, A):
        return tf.math.argmin(tf.math.reduce_sum(tf.math.pow(A - xi, 2), 1))

    # Una vez tenemos el mas cercano. Calculamos en error_ la distancia de todos con respecto a A.
    # Como ya tenemos el indice del mas cercano y tf.math.argmin() nos daría eso como el menor.
    # Asiganmos el valor infinito (np.Inf) al más cercano, de esta forma tf.math.argmin() nos sará el siguiente
    def findNearestAndSecondNearestUnit(self, xi, A):
        distance_ = tf.constant(tf.math.reduce_sum(tf.math.pow(A - xi, 2), 1), dtype=tf.float32).numpy()
        indexNearestUnit = tf.math.argmin(tf.constant(distance_))
        distance_[indexNearestUnit] = np.Inf
        return indexNearestUnit, tf.math.argmin(tf.constant(distance_))

    def findIndexNeighbourMaxError(self, indexUnitWithMaxError_):
        index = tf.squeeze(tf.math.argmax(tf.gather(self.error_, self.N[indexUnitWithMaxError_].neighborhood)), 0)
        indexNeighbourMaxError = self.N[indexUnitWithMaxError_].neighborhood[index]
        return indexNeighbourMaxError

    def pruneA(self):
        # Pillamos los indices de aquellas que no se hayan quedado sin vecinos y con tf.gather()
        # Creamos el nuevo A, únicamente con los índices que tienen más de un vecino
        indexToNotRemove = [index for index in tf.range(self.N.__len__()) if self.N[index].neighborhood.__len__() > 0]
        self.A = tf.Variable(tf.gather(self.A, indexToNotRemove, axis=0))

        # Ahora actualizamos la id de los restantes grafos para no tener problemas con las iteraciones más adelante
        # Haciendo que pivoten entre si y cojan las id de aquellas que tenian delante si estas han desaparecido.
        for graphIndex in reversed(range(self.N.__len__())):
            if self.N[graphIndex].neighborhood.__len__() == 0:
                for pivot in range(graphIndex + 1, self.N.__len__()):
                    self.N[pivot].id -= 1
                    for indexN in range(self.N.__len__()):
                        for indexNeighbothood in range(self.N[indexN].neighborhood.__len__()):
                            if self.N[indexN].neighborhood[indexNeighbothood] == pivot:
                                self.N[indexN].neighborhood[indexNeighbothood] -= 1
                self.N.pop(graphIndex)

    def getGraphConnectedComponents(self):
        connectedComponentIndeces = list(range(self.N.__len__()))
        for graphIndex in range(self.N.__len__()):
            for neighbourIndex in self.N[graphIndex].neighborhood:
                if connectedComponentIndeces[graphIndex] <= connectedComponentIndeces[neighbourIndex]:
                    connectedComponentIndeces[neighbourIndex] = connectedComponentIndeces[graphIndex]
                else:
                    aux = connectedComponentIndeces[graphIndex]
                    for pivot in range(graphIndex, self.N.__len__()):
                        if connectedComponentIndeces[pivot] == aux:
                            connectedComponentIndeces[pivot] = connectedComponentIndeces[neighbourIndex]
        uniqueConnectedComponentIndeces = functools.reduce(lambda cCI, index: cCI.append(index) or cCI if index not in cCI else cCI, connectedComponentIndeces, [])
        connectedComponents = []
        for connectedComponentIndex in uniqueConnectedComponentIndeces:
            connectedComponent = []
            for index in range(connectedComponentIndeces.__len__()):
                if connectedComponentIndex == connectedComponentIndeces[index]:
                    connectedComponent.append(self.N[index])
            connectedComponents.append(connectedComponent)
        return uniqueConnectedComponentIndeces.__len__(), connectedComponents

    def componentesConexas(self, N):
        componentes = [np.array([grafo.id] + grafo.neighborhood) for grafo in N]
        combined = [[] for _ in N]

        modified, added = [],  True

        while added:

            added = False

            for i, grupo in enumerate(componentes):
                for j, sig_grupo in enumerate(componentes[i+1:]):
                    if len(np.intersect1d(grupo, sig_grupo)) > 0 and (j + i + 1) not in combined[i]:
                        componentes[i] = np.union1d(grupo, sig_grupo)
                        combined[i].append(j + i + 1)
                        modified.append(j + i + 1)
                        added = True

        modified = list(set(modified))
        modified.sort(reverse=True)

        for i in modified:
            componentes.pop(i)


        return [c.tolist() for c in componentes], len(componentes)

    def fit(self, trainingX, numberEpochs, numberGroupMax, path):
        # Paso 1
        self.A = tf.Variable(tf.random.normal([2, trainingX.shape[1]], 0.0, 1.0, dtype=tf.float32))
        self.N.append(Graph(0))
        self.N.append(Graph(1))
        self.error_ = tf.Variable(tf.zeros([2, 1]), dtype=tf.float32)
        numberGroups = 0
        epoch = 0
        numberProcessedRow = 0
        iteration = 1
        while epoch < numberEpochs and numberGroups <= numberGroupMax:
                # Paso 2
                shuffledTrainingX = tf.random.shuffle(trainingX)
                for row_ in tf.range(shuffledTrainingX.shape[0]):


                    xi = shuffledTrainingX[row_]

                    # Paso 3
                    indexNearestUnit, indexSecondNearestUnit = self.findNearestAndSecondNearestUnit(xi, self.A)

                    # Paso 4
                    self.incrementAgeNeighborhood(indexNearestUnit)

                    # Paso 5
                    self.error_[indexNearestUnit].assign(self.error_[indexNearestUnit] + tf.math.reduce_sum(tf.math.squared_difference(xi, self.A[indexNearestUnit])))

                    # Paso 6
                    self.A[indexNearestUnit].assign(self.A[indexNearestUnit] + self.epsilon_a * (xi - self.A[indexNearestUnit]))
                    for indexNeighbour in self.N[indexNearestUnit].neighborhood:
                        self.A[indexNeighbour].assign(self.A[indexNeighbour] + self.epsilon_n * (xi - self.A[indexNeighbour]))

                    # Paso 7
                    if indexSecondNearestUnit in self.N[indexNearestUnit].neighborhood:
                        self.N[indexNearestUnit].setAge(indexSecondNearestUnit, 0.0)
                        self.N[indexSecondNearestUnit].setAge(indexNearestUnit, 0.0)
                    else:
                        self.N[indexNearestUnit].addNeighbour(indexSecondNearestUnit, 0.0)
                        self.N[indexSecondNearestUnit].addNeighbour(indexNearestUnit, 0.0)

                    # Paso 8
                    # Primero en N quitamos las conexiones que sean mayores que a_max
                    for graph in self.N:
                        graph.pruneGraph(self.a_max)

                    self.pruneA()

                    # Esto es para generar imágenes
                    numberGraphConnectedComponents, numberGroups = self.componentesConexas(self.N)

                    print("numberIterations: {} - numberUnits: {} - numberGroups: {}".format(iteration,self.A.shape[0], numberGroups))
                    iteration = iteration + 1
                    if(iteration % 2):
                        GrowingNeuralGasPlotter.plotGraphConnectedComponent(path+"/",
                                                                        'graphConnectedComponents_' + '{}_{}'.format(
                                                                            self.A.shape[0],
                                                                            numberGroups),
                                                                        self.A,
                                                                        self.N)

                    # Paso 9
                    if not (numberProcessedRow + 1) % self.eta:
                        # Escogemos la unidad de mayor error y su vecina
                        indexUnitWithMaxError_ = tf.squeeze(tf.math.argmax(self.error_), 0)
                        indexNeighbourWithMaxError_ = self.findIndexNeighbourMaxError(indexUnitWithMaxError_)

                        # Añadimos esta nueva variable entre las dos de mayor error
                        self.A = tf.Variable(tf.concat([self.A, tf.expand_dims(0.5 * (self.A[indexUnitWithMaxError_] + self.A[indexNeighbourWithMaxError_]), 0)], 0))

                        self.N.append(Graph(self.A.shape[0] - 1, [indexUnitWithMaxError_, indexNeighbourWithMaxError_], [0.0, 0.0]))
                        # Quitamos la arista que une las  de mayor error y las unimos a trvés de la nueva
                        self.N[indexUnitWithMaxError_].removeNeighbour(indexNeighbourWithMaxError_)
                        self.N[indexUnitWithMaxError_].addNeighbour(tf.constant(self.A.shape[0] - 1, dtype=tf.int64), 0.0)
                        self.N[indexNeighbourWithMaxError_].removeNeighbour(indexUnitWithMaxError_)
                        self.N[indexNeighbourWithMaxError_].addNeighbour(tf.constant(self.A.shape[0] - 1, dtype=tf.int64), 0.0)

                        # Decrementamos el error de las unidades por alpha
                        self.error_[indexUnitWithMaxError_].assign(self.error_[indexUnitWithMaxError_] * self.alpha)
                        self.error_[indexNeighbourWithMaxError_].assign(self.error_[indexNeighbourWithMaxError_] * self.alpha)
                        # Añadimos a la variable error la nueva unidad con el error igual al de la unidad con mayor error (escogida antes)
                        self.error_ = tf.Variable(tf.concat([self.error_,  tf.expand_dims(self.error_[indexUnitWithMaxError_], 0)], 0))

                    # Paso 10
                    self.error_.assign(self.error_ * self.delta)
                    # contamos cuantas xi han sido procesadas
                    numberProcessedRow += 1

                epoch += 1
                print("GrowingNeuralGas::epoch: {}".format(epoch))


    def predict(self, X):
        agrupamientos, _ = self.componentesConexas(self.N)

        predictions = []
        for x in X:
            indexNearestUnit = self.findNearestUnit(x, self.A)

            for grupo in agrupamientos:
                if indexNearestUnit in grupo:
                    predictions.append(agrupamientos.index(grupo))
                    break

        return predictions

    def saver(self, dirpath):

        pack = [
            self.epsilon_a,
            self.epsilon_n,
            self.a_max,
            self.eta,
            self.alpha,
            self.delta,
            self.maxNumberUnits
        ]
        GNGSaver = GrowingNeuralGasSaver()
        GNGSaver.saveGNG(self.A, self.N, self.error_, dirpath, pack)

    def loader(self, dirpath):
        GNGLoader = GrowingNeuralGasLoader()
        self.A, self.N, self.error_, pack = GNGLoader.loadGNG(dirpath)
        self.epsilon_a = pack[0]
        self.epsilon_n = pack[1]
        self.a_max = pack[2]
        self.eta = pack[3]
        self.alpha = pack[4]
        self.delta = pack[5]
        self.maxNumberUnits = pack[6]