import matplotlib.pyplot
import tensorflow as tf

from matplotlib import colors

class GrowingNeuralGasPlotter(object):
    @staticmethod
    def plotGraphConnectedComponent(pathFigure, nameFigure, A, N):
        figure, axis = matplotlib.pyplot.subplots()

        for index in tf.range(N.__len__()):
            for neighbourIndex in tf.range(N[index].neighborhood.__len__()):
                axis.plot([A[index][0].numpy(), A[N[index].neighborhood[neighbourIndex]][0].numpy()],[A[index][1].numpy(), A[N[index].neighborhood[neighbourIndex]][1].numpy()],"k.-")


        figure.savefig(pathFigure + nameFigure + '.png', transparent=False, dpi=80, bbox_inches="tight")
        matplotlib.pyplot.close(figure)