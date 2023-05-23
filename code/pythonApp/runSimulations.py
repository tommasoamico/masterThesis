from modules.utilityFunctions import inverseCum, brackets, cdfLogSpacePositiveH, cdfLogSpace
from modules.constantsAndVectors import gammaValues
import numpy as np
from typing import List, Type
from scipy.optimize import root_scalar
from tqdm import tqdm
from modules.constantsAndVectors import gammaValues
from modules.simulations import simulation
import matplotlib.pyplot as plt

simulation.all: List[Type[simulation]] = []

seriesLength: int = 250

simulation.instantiateFromIterable(
    gammaValues=gammaValues, seriesLength=seriesLength)

arraySizes = np.zeros((len(gammaValues), seriesLength))
arrayAcorr = np.zeros((len(gammaValues), seriesLength))

for i, instance in enumerate(tqdm(simulation.all)):
    sizes, acorr = instance.simulate()
    arraySizes[i, :] = sizes
    arrayAcorr[i, :] = acorr

savePath: str = '/Users/tommaso/Desktop/masterThesis/data/nullHShortLineages/'

np.save(savePath + 'gammaValues.npy', gammaValues)
np.save(savePath + 'timeSerieses.npy', arraySizes)
np.save(savePath + 'autoCorrelations.npy', arrayAcorr)
