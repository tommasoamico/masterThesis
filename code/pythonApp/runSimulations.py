from modules.utilityFunctions import inverseCum, brackets, cdfLogSpacePositiveH, cdfLogSpace
from modules.constantsAndVectors import hValueses, gammaPositiveH
import numpy as np
from typing import List, Type
from scipy.optimize import root_scalar
from tqdm import tqdm
from modules.constantsAndVectors import gammaValues
from modules.simulations import simulation
import matplotlib.pyplot as plt
import os


simulation.all: List[Type[simulation]] = []

seriesLength: int = int(1e6)
valuesToSave: int = int(5e5)
savePath1: str = '/Users/tommaso/Desktop/masterThesis/data/calibratedModel/studentsH/'

assert os.path.exists(savePath1), "Save path does not exists"
# savePathShort: str = '/Users/tommaso/Desktop/masterThesis/data/positiveH/activeHHDecidingShort/'


simulation.instantiateFromIterableHs(
    gamma=gammaPositiveH, seriesLength=seriesLength, hValues=hValueses)

arraySizes = np.zeros((len(hValueses), seriesLength))
arrayAcorr = np.zeros((len(hValueses), seriesLength))

for i, instance in enumerate(tqdm(simulation.all)):
    sizes, acorr = instance.simulateProtein()
    arraySizes[i, :] = sizes
    arrayAcorr[i, :] = acorr
print(np.mean(arraySizes[0, :]))
# np.save(savePath1 + 'hValues.npy', hValueses)

np.save(savePath1 + 'timeSerieses.npy', arraySizes[:, -valuesToSave:])
# np.save(savePathShort + 'timeSerieses.npy', arraySizes[:, -valuesToSaveShort:])
np.save(savePath1 + 'autoCorrelations.npy', arrayAcorr)
np.save(savePath1 + 'hValueses.npy', arrayAcorr)
