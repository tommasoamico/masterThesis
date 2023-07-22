import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from pprint import pprint
from statsmodels.graphics.tsaplots import plot_predict
from modules.markovChainModule import markovChain
from modules.utilityFunctions import fitFrequentist, errorPropagation1, errorPropagation2
from functools import reduce

basePath = '/Users/tommaso/Desktop/masterThesis/data/realData/Susman/longLineages/'
basePath: str = '/Users/tommaso/Desktop/masterThesis/data/realData/Tanouchi/Tanouchi37/'
dataPath = basePath + 'Tanouchi37C.csv'
markovChain.all = []
markovChain.instantiateFromCsvRD(dataPath)

allIdxs = list(map(lambda x: x.filterOutliers(), markovChain.all))

allCoefficients = [fitFrequentist(xData=markovChain.all[i].timeSeries.to_numpy(),
                                  yData=markovChain.all[i].lengthFinal.to_numpy(
),
    outliersIdx=allIdxs[i]) for i in range(len(markovChain.all))]

allAcorrMethod1 = list(
    map(lambda x: (-np.log(x[0]), errorPropagation1(x[0], x[1])), allCoefficients))

allAcorrMethod2 = list(map(
    lambda x: (-1 / np.log(x[0]), errorPropagation2(x[0], x[1])), allCoefficients))

aCorrValues1 = list(map(lambda x: x[0], allAcorrMethod1))
aCorrValues2 = list(map(lambda x: x[0], allAcorrMethod2))

errorValues1 = list(map(lambda x: x[1], allAcorrMethod1))
errorValues2 = list(map(lambda x: x[1], allAcorrMethod2))

allMeans = np.array([np.delete(markovChain.all[i].timeSeries.to_numpy(), [
                    allIdxs[i]]).mean() for i in range(len(markovChain.all))])

# print(
#    list(map(lambda x: (-1 / np.log(x[0]), errorPropagation2(x[0], x[1])), allCoefficients)))
tanouchiMethod1 = np.vstack(
    [np.array(aCorrValues1), np.array(errorValues1), allMeans])
tanouchiMethod2 = np.vstack(
    [np.array(aCorrValues2), np.array(errorValues2), allMeans])

np.save(basePath + 'tanouchiCorrelations1.npy', tanouchiMethod1)
np.save(basePath + 'tanouchiCorrelations2.npy', tanouchiMethod2)

# print(exampleInstance.timeSeries)
# print(exampleInstance.growthRate)
# print(exampleInstance.generationTime)
# print(exampleInstance.meanLineage(timeSeries,
#      exampleInstance.growthRate, exampleInstance.generationTime))


# plt.scatter(range(len(exampleInstance.timeSeries)), exampleInstance.timeSeries)
# plt.scatter(range(len(exampleInstance.timeSeries)),
#  exampleInstance.lengthFinal)
# plt.show()

# print([x.shape for x in paddedTS])


'''
for i, order in enumerate(arimaParameters):
aic: float = arima.getAic(order=order)
allAic[i] = aic
minIdx: int = np.argmin(allAic)
minIdx = 5
parametersMin: Tuple[int, int, int] = arimaParameters[minIdx]
prediction: np.ndarray = arima.getPrediciton(
parametersMin, lenPrediction=1000)
plt.plot(prediction)
plt.show()
'''
