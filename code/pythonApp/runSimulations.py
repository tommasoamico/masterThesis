from modules.utilityFunctions import inverseCum, brackets, cdfLogSpacePositiveH, cdfLogSpace
from modules.constantsAndVectors import hValueses, gammaPositiveH
import numpy as np
from typing import List, Type
from scipy.optimize import root_scalar
from tqdm import tqdm
from modules.constantsAndVectors import gammaValues
from modules.simulations import simulation
import matplotlib.pyplot as plt

simulation.all: List[Type[simulation]] = []

seriesLength: int = int(2e6)
valuesToSave: int = int(1e6)
valuesToSaveShort = 250
savePath1: str = '/Users/tommaso/Desktop/masterThesis/data/positiveH/absorbingHHDeciding/'
savePathShort: str = '/Users/tommaso/Desktop/masterThesis/data/positiveH/absorbingHHDecidingShort/'


simulation.instantiateFromIterableHs(
    gamma=gammaPositiveH, seriesLength=seriesLength, hValues=hValueses)

arraySizes = np.zeros((len(hValueses), seriesLength))
arrayAcorr = np.zeros((len(hValueses), seriesLength))

for i, instance in enumerate(tqdm(simulation.all)):
    sizes, acorr = instance.simulate()
    arraySizes[i, :] = sizes
    arrayAcorr[i, :] = acorr


np.save(savePath1 + 'hValues.npy', hValueses)
np.save(savePathShort + 'hValues.npy', hValueses)
np.save(savePath1 + 'timeSerieses.npy', arraySizes[:, -valuesToSave:])
np.save(savePathShort + 'timeSerieses.npy', arraySizes[:, -valuesToSaveShort:])
np.save(savePath1 + 'autoCorrelations.npy', arrayAcorr)
