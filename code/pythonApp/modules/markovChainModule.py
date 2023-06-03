import numpy as np
import pandas as pd
from typing import List, Tuple, Union
from statsmodels.tsa.stattools import acf


class markovChain:
    all = []

    def __init__(self, timeSeries: np.array, idNumber: int = 1) -> None:
        self.timeSeries = timeSeries
        self.idNumber = idNumber

        markovChain.all.append(self)

    def __len__(self) -> int:
        return len(self.timeSeries)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__} instance, id = {self.idNumber}'

    def __enter__(self) -> None:
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

    @classmethod
    def instantiateFromCsv(cls, pathCsv: str) -> None:
        df: pd.DataFrame = pd.read_csv(pathCsv)
        df['lineage_ID'] = df['lineage_ID'].astype(int)
        uniqueLineages: np.array = df['lineage_ID'].unique()
        for lineage in uniqueLineages:
            newDf: pd.DataFrame = df[df['lineage_ID'] == lineage]
            markovChain(np.array(newDf['length_birth']), idNumber=lineage)

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

    @classmethod
    def instantiateFromNpy(cls, npyPath: str, log: bool = False) -> None:

        matrix: np.array = np.load(npyPath)

        for i in range(matrix.shape[0]):
            if log:
                markovChain(np.exp(matrix[i, :]), idNumber=i + 1)
            else:
                markovChain(matrix[i, :], idNumber=i + 1)
