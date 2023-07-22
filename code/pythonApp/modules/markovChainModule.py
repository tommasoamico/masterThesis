import numpy as np
import pandas as pd
from typing import List, Tuple, Union, Optional, Type
from statsmodels.tsa.stattools import acf
from tqdm import tqdm
from typeguard import typechecked


class markovChain:
    all = []

    def __init__(self, timeSeries: np.ndarray, idNumber: int = 1, lengthFinal: Optional[np.ndarray] = None, growthRate: Optional[np.ndarray] = None) -> None:
        self.timeSeries = timeSeries
        self.idNumber = idNumber
        if lengthFinal is not None:
            self.lengthFinal = lengthFinal
        if growthRate is not None:
            self.growthRate = growthRate
        if lengthFinal is not None and growthRate is not None:
            self.generationTime = (1 / self.growthRate) * \
                np.log((self.lengthFinal / self.timeSeries))
            self.meanLineages = self.meanLineage(
                lengthBirth=self.timeSeries, growthRate=self.growthRate, generationTime=self.generationTime)

            self.varianceLineages = self.varianceLineage(
                lengthBirth=self.timeSeries, growthRate=self.growthRate, generationTime=self.generationTime)

            self.finalMean = (np.sum(self.meanLineages *
                                     self.generationTime)) / np.sum(self.generationTime)

            # print(self.varianceLineages)
            self.finalStd = np.sqrt((np.sum(self.varianceLineages *
                                            self.generationTime)) / np.sum(self.generationTime))

        markovChain.all.append(self)

    @staticmethod
    def meanLineage(lengthBirth: Union[np.ndarray, float], growthRate: Union[np.ndarray, float], generationTime: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
        return (1 / generationTime) * (lengthBirth / growthRate) * (np.exp(growthRate * generationTime) - 1)

    @staticmethod
    def varianceLineage(lengthBirth: Union[np.ndarray, float], growthRate: Union[np.ndarray, float], generationTime: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
        x0 = lengthBirth
        gr = growthRate
        tau = generationTime
        std = ((x0**2) / (2 * gr)) * (np.exp(2 * gr * tau) - 1) + (markovChain.meanLineage(x0, gr, tau)
                                                                   ** 2)*tau - 2*(markovChain.meanLineage(x0, gr, tau)/gr) * x0 * (np.exp(gr * tau) - 1)
        # meanLin: Union[float, np.ndarray] = markovChain.meanLineage(**locals())

        return (1 / tau) * std

    def __len__(self) -> int:
        return len(self.timeSeries)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__} instance, id = {self.idNumber}'

    def __enter__(self) -> Type["markovChain"]:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        print('Exiting...')

    def meanSizeAtBirth(self) -> float:
        return np.mean(self.timeSeries)

    def kthMoment(self, k: float) -> float:
        return np.mean(self.timeSeries**k)

    def momentRatio(self, k: float) -> float:
        moment1: float = self.kthMoment(self.timeSeries, k + 1)
        moment2: float = self.kthMoment(self.timeSeries, k)
        return moment1 / moment2

    def padTS(self, nToAdd: int) -> np.ndarray:
        factor: int = 5
        pointToSample: int = nToAdd // factor
        vectorToReturn: np.ndarray = self.timeSeries[:69].copy()
        if len(vectorToReturn) < 69:
            vectorToReturn: np.ndarray = np.pad(
                vectorToReturn, pad_width=(0, 1), mode='mean')
        paddingVector: np.ndarray = np.zeros(pointToSample * factor)

        for i in range(pointToSample):
            index: int = np.random.randint(
                low=0, high=len(self.timeSeries) - (factor - 1))
            triplets: np.ndarray = self.timeSeries[index:index + factor]
            paddingVector[factor*i:factor*(i + 1)] = triplets
        vectorToReturn: np.ndarray = np.concatenate(
            [vectorToReturn, paddingVector])
        return vectorToReturn

    @classmethod
    def instantiateFromCsvRD(cls, pathCsv: str,) -> None:
        df: pd.DataFrame = pd.read_csv(pathCsv)
        df['lineage_ID'] = df['lineage_ID'].astype(int)
        uniqueLineages: np.array = df['lineage_ID'].unique()
        for lineage in uniqueLineages:
            newDf: pd.DataFrame = df[df['lineage_ID'] == lineage]
            timeSeries: pd.Series = newDf['length_birth']
            lengthFinal: pd.Series = newDf['length_final']
            growthRate: pd.Series = newDf['growth_rate']

            markovChain(timeSeries=timeSeries, idNumber=lineage,
                        lengthFinal=lengthFinal, growthRate=growthRate)

    def autocorrelation(self, maxLags: int) -> np.array:
        autocorrelation = acf(self.timeSeries, nlags=maxLags - 1, fft=True)
        return autocorrelation

    def pdf(self, bins: Union[str, int]) -> Tuple[np.array]:
        counts, edges = np.histogram(self.timeSeries, bins=bins, density=True)
        binCenters = .5*(edges[1:] + edges[:-1])
        return counts, binCenters

    def survival_data(self) -> Tuple[np.array]:
        array = np.sort(self.timeSeries)
        array = array[~np.isnan(array)]
        array = array[array >= 0]
        cumul = 1 - np.arange(0, len(array))/(len(array))
        return array, cumul

    @typechecked
    def filterOutliers(self) -> np.ndarray:
        firstCriteria: np.ndarray = np.where(
            self.timeSeries > self.finalMean + 2 * self.finalStd)[0]
        secondCriteria: np.ndarray = np.where(
            self.lengthFinal > 2 * (self.finalMean + 2 * self.finalStd))[0]
        thirdCriteria: np.ndarray = np.where(
            self.timeSeries < self.finalMean - 2 * self.finalStd)[0]
        fourthCriteria: np.ndarray = np.where(
            self.timeSeries > self.lengthFinal)[0]

        allIdx: np.ndarray = np.concatenate(
            [firstCriteria, secondCriteria, thirdCriteria, fourthCriteria])

        return allIdx

    @classmethod
    def instantiateFromNpy(cls, npyPath: str, log: bool = False) -> None:

        matrix: np.array = np.load(npyPath)

        for i in range(matrix.shape[0]):
            if log:
                markovChain(np.exp(matrix[i, :]), idNumber=i + 1)
            else:
                markovChain(matrix[i, :], idNumber=i + 1)

    @classmethod
    def instantiateFromCsv(cls, csvPath: str, log: bool = False, limit: Optional[str] = None) -> None:

        matrix: np.ndarray = np.loadtxt(csvPath, delimiter=',')

        if limit is not None:
            matrix: np.ndarray = matrix[:, limit:]

        for i in range(matrix.shape[0]):
            if log:
                markovChain(np.exp(matrix[i, :]), idNumber=i + 1)
            else:
                markovChain(matrix[i, :], idNumber=i + 1)
