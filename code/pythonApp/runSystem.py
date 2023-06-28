from modules.visualizationTS import systemHandling
from modules.constantsAndVectors import slopeSimulation2, interceptSimulation2, slopeSimulation3, interceptSimulation3
import numpy as np
from pathlib import Path
import json
from typing import List
from scipy.stats import linregress
import pandas as pd
from functools import reduce
import matplotlib.pyplot as plt

stawskiPath = '/Users/tommaso/Desktop/masterThesis/data/realData/Susman/longLineages/'
allMeansStawski = np.load(stawskiPath + 'momentScaling/allMeans.npy')
allMomentsStawski = np.load(stawskiPath + 'momentScaling/allMoments.npy')

tanouchi25Path = '/Users/tommaso/Desktop/masterThesis/data/realData/Tanouchi/Tanouchi25/'
allMeansTanouchi25 = np.load(tanouchi25Path + 'momentScaling/allMeans.npy')
allMomentsTanouchi25 = np.load(tanouchi25Path + 'momentScaling/allMoments.npy')

tanouchi37Path = '/Users/tommaso/Desktop/masterThesis/data/realData/Tanouchi/Tanouchi37/'
allMeansTanouchi37 = np.load(tanouchi37Path + 'momentScaling/allMeans.npy')
allMomentsTanouchi37 = np.load(tanouchi37Path + 'momentScaling/allMoments.npy')

dataset = 'tanouchi37'


outliersTanouchi37: np.array = np.where(allMomentsTanouchi37[1] > 20)[0]

tanouchi37FinalMoments = np.delete(
    allMomentsTanouchi37, outliersTanouchi37, axis=1)
tanouchi37FinalMeans = np.delete(
    allMeansTanouchi37, outliersTanouchi37, axis=1)

datasetDict = {'allMeans': {'Stawski': allMeansStawski, 'tanouchi25': allMeansTanouchi25, 'tanouchi37': tanouchi37FinalMeans},
               'allMoments': {'Stawski': allMomentsStawski, 'tanouchi25': allMomentsTanouchi25, 'tanouchi37': tanouchi37FinalMoments},
               'slopeRange': {'Stawski': {'2': np.linspace(1.9, 2.1, 400), '3': np.linspace(2.6, 2.9, 400)}, 'tanouchi25': {'2': np.linspace(1.7, 2.3, 400), '3': np.linspace(2.7, 3.7, 400)}, 'tanouchi37': {'2': np.linspace(1.7, 2.5, 400), '3': np.linspace(2.6, 4.1, 400)}},
               'interceptRange': {'Stawski': {'2': np.linspace(0, .2, 400), '3': np.linspace(.1, .3, 400)}, 'tanouchi25': {'2': np.linspace(-.2, .2, 400), '3': np.linspace(-.2, .4, 400)}, 'tanouchi37': {'2': np.linspace(-.2, .2, 400), '3': np.linspace(-.4, .4, 400)}},
               }

scatterData = {'slope': {2: slopeSimulation2, 3: slopeSimulation3},
               'intercept': {2: interceptSimulation2, 3: interceptSimulation3}}
print(slopeSimulation3, interceptSimulation3)
dataPath: Path = Path.cwd().parents[1] / 'data'

kIdx: int = 1
momentNumber: int = kIdx + 2
with systemHandling(allMeans=datasetDict['allMeans'][dataset], allMoments=datasetDict['allMoments'][dataset]) as system:
    resultFit = linregress(
        np.log10(system.allMeans[0, :]), np.log10(system.allMoments[kIdx, :]))
    slopeRange = datasetDict['slopeRange'][dataset][str(momentNumber)]
    interceptRange = datasetDict['interceptRange'][dataset][str(momentNumber)]
    system.lrLandscape(slopeRange=slopeRange, interceptRange=interceptRange, show=False, scatterPoints=[(
        resultFit.slope, resultFit.intercept, f'k = {momentNumber} data'), (scatterData['slope'][momentNumber], scatterData['intercept'][momentNumber], f'k={momentNumber} simulation')], title=dataset, kIdx=kIdx)
    print(resultFit.slope, resultFit.intercept)
    # slopeRange: np.ndarray = np.linspace(
    #    1.7, 2.2, 400)
    # interceptRange: np.ndarray = np.linspace(0, .2, 400)
    # systemHandling.allInstances = []
    # systemHandling.instantiateFromList(
    # listMeans=[allMeansStawski, allMeansTanouchi25, allMeansTanouchi37], listMoments=[allMomentsStawski, allMomentsTanouchi25, allMomentsTanouchi37])
# allGridk2 = list(map(lambda x: x.getHeatmapGrid(slopeRange=slopeRange,
 #                interceptRange=interceptRange, kIdx=0), systemHandling.allInstances))

    tickSeparation: int = len(slopeRange) // 10
# grid2Summed: np.ndarray = reduce(lambda x, y: x + y, allGridk2)
# grid2Summed /= np.max(grid2Summed)
    # plt.imshow(grid2Summed, cmap='hot', interpolation='nearest')
    # plt.colorbar()

    plt.xticks(range(len(slopeRange))[::tickSeparation], labels=list(
        map(lambda x: round(x, 2), slopeRange[::tickSeparation])))
    plt.yticks(range(len(interceptRange))[::tickSeparation], list(
        map(lambda x: round(x, 2), interceptRange[::tickSeparation])))

    plt.title(f'MSE landscape, {dataset}', fontsize=18)
    plt.xlabel('Slopes', fontsize=15)
    plt.ylabel('Intercepts', fontsize=15)
    plt.legend()
    plt.savefig(f'/Users/tommaso/Desktop/{dataset}{momentNumber}.png', dpi=300)

plt.show()
