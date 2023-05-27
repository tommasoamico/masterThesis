import numpy as np
from modules.markovChainModule import markovChain
import matplotlib.pyplot as plt
from typing import List, Type
from modules.utilityFunctions import stackList

sizesPath: str = '/Users/tommaso/Desktop/masterThesis/data/positiveH/absorbingHDeltaDecidingShort/timeSerieses.npy'
savePath: str = '/Users/tommaso/Desktop/masterThesis/data/positiveH/absorbingHDeltaDecidingShort/'
markovChain.all: List[Type[markovChain]] = []
markovChain.instantiateFromNpy(sizesPath, log=True)
kValues: np.array = np.arange(2, 6)

allMoments: List[list] = [list(map(lambda instance: instance.kthMoment(k), markovChain.all))
                          for k in kValues]

allMeans: List[list] = [list(map(lambda x: x.meanSizeAtBirth(),
                                 markovChain.all))] * len(kValues)


np.save(savePath + 'allMeans.npy', stackList(allMeans))
np.save(savePath + 'allMoments.npy', stackList(allMoments))
np.save(savePath + 'kValues.npy', stackList(kValues))
