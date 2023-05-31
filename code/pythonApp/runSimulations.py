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

seriesLength: int = int(8e6)

savePath: str = '/Users/tommaso/Desktop/masterThesis/data/nullH/nullH8MilionSamples/'

gammaValues: np.array = np.load(savePath + 'gammaValues.npy')

simulation.instantiateFromIterableGammas(
    gammaValues=gammaValues, seriesLength=seriesLength, h=0)

arraySizes = np.zeros((len(gammaValues), seriesLength))
arrayAcorr = np.zeros((len(gammaValues), seriesLength))

for i, instance in enumerate(tqdm(simulation.all)):
    sizes, acorr = instance.simulate()
    arraySizes[i, :] = sizes
    arrayAcorr[i, :] = acorr




# np.save(savePath + 'gammaValues.npy', hValueses)
np.save(savePath + 'timeSerieses.npy', arraySizes)
np.save(savePath + 'autoCorrelations.npy', arrayAcorr)
