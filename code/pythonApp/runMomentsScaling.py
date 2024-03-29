import numpy as np
from modules.markovChainModule import markovChain
import matplotlib.pyplot as plt
from typing import List, Type, Union
from modules.utilityFunctions import stackList
from pathlib import Path
from scipy.stats import linregress
from tqdm import tqdm

dataPath: Union[str, Path] = Path.cwd().parents[1] / 'data'
# /Users/tommaso/Desktop/masterThesis/data/ML/timeGan/tanouchi25/sampledDataReshaped.npy

sizesPath: Union[str, Path] = dataPath /  \
    'calibratedModel' / 'bayesianSimTanouchi37ThirdInstance' / 'timeSerieses.npy'

savePath: Union[str, Path] = dataPath / \
    'calibratedModel' / 'bayesianSimTanouchi37ThirdInstance' / 'momentScaling'
# limit = 20000
markovChain.all: List[Type[markovChain]] = []
markovChain.instantiateFromNpy(sizesPath, log=False)
kValues: np.array = np.arange(2, 6)

allMoments: List[list] = [list(map(lambda instance: instance.kthMoment(k), markovChain.all))
                          for k in kValues]

allMeans: List[list] = [list(map(lambda x: x.meanSizeAtBirth(),
                                 markovChain.all))] * len(kValues)


with open(savePath / 'momentsFit.csv', mode='w') as f:
    f.write('slope,intercept,k\n')
    for i in tqdm(range(stackList(allMeans).shape[0])):

        slope, intercept, _, _, _ = linregress(
            stackList(allMeans)[i, :], stackList(allMoments)[i, :])
        f.write(f'{slope},{intercept},{kValues[i]}\n')


np.save(savePath / 'allMeans.npy', stackList(allMeans))
np.save(savePath / 'allMoments.npy', stackList(allMoments))
np.save(savePath / 'kValues.npy', stackList(kValues))
