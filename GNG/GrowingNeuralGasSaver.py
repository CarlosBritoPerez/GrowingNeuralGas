import numpy as np
import os
import json

class GrowingNeuralGasSaver(object):

    def saveGNG(self, A, N, error_,  dirpath, pack):
        n = 0

        if not (os.path.exists(dirpath)):
            os.mkdir(dirpath)

        np.savetxt(dirpath + "/A_save.csv", np.asarray(A.numpy()), delimiter=",")
        np.savetxt(dirpath + "/error_save.csv", np.asarray(error_.numpy()), delimiter=",")

        with open(dirpath + "/N.json", "wt") as file:
            file.write(json.dumps([[i.ageNeighborhood, i.id, [j.numpy().tolist() for j in i.neighborhood]] for i in N]))

        with open(dirpath + "/variables.json", "wt") as file:
            file.write(json.dumps(pack))