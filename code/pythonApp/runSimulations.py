from modules.utilityFunctions import inverseCum, brackets, cdfLogSpacePositiveH, cdfLogSpace
from modules.constantsAndVectors import tanouchi37Bayes
import numpy as np
from typing import List, Type
from scipy.optimize import root_scalar
from tqdm import tqdm
from modules.constantsAndVectors import gammaValues
from modules.simulations import simulation
import matplotlib.pyplot as plt
import pandas as pd
import os
import time

simulation.all: List[Type[simulation]] = []

seriesLength: int = int(4e5)
valuesToSave: int = int(3e5)
savePath1: str = '/Users/tommaso/Desktop/masterThesis/data/calibratedModel/bayesianSimTanouchi37ThirdInstance/'

assert os.path.exists(savePath1), "Save path does not exists"
# savePathShort: str = '/Users/tommaso/Desktop/masterThesis/data/positiveH/activeHHDecidingShort/'

bayesDf: pd.DataFrame = pd.read_csv(tanouchi37Bayes)
bayesDf['h'] = bayesDf['u'] / bayesDf['v']
bayesDf['gamma'] = bayesDf['omega_2'] / ((bayesDf['c']) * bayesDf['d'])

simulation.instantiateFromIterableHs(
    hValues=bayesDf['h'].to_numpy(), seriesLength=seriesLength, gamma=1.5)

arraySizes = np.zeros((len(bayesDf), seriesLength))
arrayAcorr = np.zeros((len(bayesDf), seriesLength))
# add tqdm
'''
h = 8
sim = simulation(gamma=gammaPositiveH, seriesLength=seriesLength, h=h)
start = time.time()
sim: Type[simulation]
sim.simulate()
end = time.time()
'''

for i, instance in enumerate(tqdm(simulation.all)):

    sizes, acorr = instance.simulateProtein()

    arraySizes[i, :] = sizes
    arrayAcorr[i, :] = acorr
# print(np.mean(arraySizes[0, :]))

# np.save(savePath1 + 'hValues.npy', hValueses)

np.save(savePath1 + 'timeSerieses.npy',
        arraySizes[:, -valuesToSave:])
# np.save(savePathShort + 'timeSerieses.npy', arraySizes[:, -valuesToSaveShort:])
np.save(savePath1 + 'autoCorrelations.npy', arrayAcorr)
# np.save(savePath1 + 'hValueses.npy', arrayAcorr)
