import numpy as np
from typing import Tuple, List, Type
from statsmodels.tsa.arima.model import ARIMA
from typeguard import typechecked
import matplotlib.pyplot as plt
from tqdm import tqdm


class arimaHandler:

    all: List["arimaHandler"] = []

    def __init__(self, timeseries: np.ndarray) -> None:
        assert len(timeseries.shape) == 1, 'Array must be 1 dimensional'
        self.timeseries = timeseries

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        print('Exiting...')

    @typechecked
    def getAic(self, order: Tuple[int, int, int]) -> float:
        model = ARIMA(self.timeseries, order=order)
        fitResult = model.fit()
        # prediction = fitResult.predict(
        #   start=len(self.timeseries), end=2 * len(self.timeseries), typ='levels')
        return fitResult.aic

    @typechecked
    def getPrediciton(self, order: Tuple[int, int, int], lenPrediction: int) -> np.ndarray:

        nIterations = lenPrediction // len(self.timeseries)
        finalPrediction = np.zeros(len(self.timeseries) * nIterations)
        for i in tqdm(range(nIterations)):
            model = ARIMA(self.timeseries, order=order)
            fitResult = model.fit()

            finalPrediction[i * len(self.timeseries): np.min([(i + 1) * len(self.timeseries), len(finalPrediction)])
                            ] = fitResult.predict(start=0, end=len(self.timeseries) - 1, typ='levels')

        # prediction: np.ndarray = fitResult.predict(
         #   start=len(self.timeseries), end=len(self.timeseries) + lenPrediction, typ='levels')
        # entirePrediction: np.ndarray = np.concatenate(
         #   [self.timeseries, prediction])

        return finalPrediction
