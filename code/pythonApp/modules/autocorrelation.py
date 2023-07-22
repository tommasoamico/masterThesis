import numpy as np
from typing import Tuple, List, Type, Optional
from statsmodels.tsa.stattools import acf
from scipy.optimize import curve_fit
from modules.utilityFunctions import rangeAutocorrelation, head2
from sklearn.metrics import r2_score
from scipy.signal import hilbert
from modules.utilityFunctions import getPeaks, tail
from scipy.interpolate import interp1d
import pandas as pd
from tqdm import tqdm


class autoCorrelation:
    all: list = []

    def __init__(self, autocorrelation: np.ndarray, idNumber: int = 1) -> None:
        assert len(autocorrelation.shape) == 1, 'Array must be 1 dimensional'

        self.autocorrelation: np.ndarray = autocorrelation
        # self.lags: np.array = rangeAutocorrelation(self.autocorrelation)
        self.lags: np.ndarray = np.arange(len(self.autocorrelation))

        self.xAxis: np.ndarray = np.linspace(
            np.min(self.lags), np.max(self.lags), 1000)

        self.idNumber: int = idNumber

        autoCorrelation.all.append(self)

    def __enter__(self) -> None:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        print('Exiting...')

    def __len__(self) -> int:
        return len(self.autocorrelation)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__} instance, id = {self.idNumber}'

    def __str__(self) -> str:
        return f'{self.__class__.__name__} instance, id = {self.idNumber}'

    def getAutocorrelationFit(self, aurocorrelationFunction: callable, x: np.ndarray, y: np.array, p0=None) -> Tuple[np.array, np.array, np.array]:
        '''
        Fits the time series with a certain autocorrelation function (aurocorrelationFunction) which must take the independent variable as the first argument,
          given a certain autocorrelation vector (autocorrelation)

        Returns:
        popt: array of fitted parameters
        pcov: covariance matrix
        yPred: the predicted autocorrelation 
        '''

        popt, pcov = curve_fit(aurocorrelationFunction,
                               x, y, p0=p0)

        yPred: np.ndarray = aurocorrelationFunction(self.lags, *popt)

        return popt, pcov, yPred

    @staticmethod
    def r2Score(yTrue: np.ndarray, yPred: np.ndarray) -> float:
        '''
        Returns the r2Score of two arrays yTrue and yPred
        '''
        r2Score: float = r2_score(y_true=yTrue, y_pred=yPred)

        return r2Score

    def getEnvelopeHilbert(self) -> np.array:
        '''
        Returns the upper envelope, retrieved thanks to the Hilbert transform
        '''
        analyticSignal: np.array[np.complex] = hilbert(self.autocorrelation)
        amplitudeEnvelope: np.array = np.abs(analyticSignal)

        return amplitudeEnvelope

    # Consider also find_peaks_cwt
    def envelopePeaks(self) -> Tuple[np.array, np.array]:
        '''
        Function that returns the indexs of upper and lower envelope of the provided autoorrelation vector
        '''
        upperPeaks = getPeaks(self.autocorrelation)

        lowerPeaks = getPeaks(-self.autocorrelation)

        return upperPeaks, lowerPeaks

    def getInterpolatingFunction(self, peaks: np.array) -> callable:
        x = self.lags
        y = self.autocorrelation[peaks]

        fLeft = interp1d(x[peaks], y, kind='cubic')
        fRight = interp1d(x[peaks], y, kind='linear')

        def f(t: float) -> float:
            if t <= x[peaks[1]] and t >= np.min(x):
                return fLeft(t)
            elif t < np.max(x) and t > x[1]:
                return fRight(t)
            else:
                return 0

        return f

    @staticmethod
    def applyInterpolatingFunction(function: callable, array: np.array) -> np.array:
        return list(map(lambda x: function(x), list(array)))

    @classmethod
    def instantiateFromCsv(cls, pathCsv: str) -> None:
        df: pd.DataFrame = pd.read_csv(pathCsv)
        df['lineage_ID'] = df['lineage_ID'].astype(int)
        uniqueLineages: np.ndarray = df['lineage_ID'].unique()
        for lineage in uniqueLineages:
            newDf: pd.DataFrame = df[df['lineage_ID'] == lineage]
            autocorrelation = acf(
                np.array(newDf['length_birth']), fft=True, nlags=len(newDf) - 1)
            autoCorrelation(autocorrelation, idNumber=lineage)

    # assume sorted in scipy interpolate

    # To intsnytiate from this

    @classmethod
    def instantiateFromTimeSeries(cls, timeSeries) -> Type:

        nlags: int = len(timeSeries) - 1

        autocorrelation: np.array = acf(timeSeries, fft=True,
                                        nlags=nlags)

        return autoCorrelation(autocorrelation)

    def getCorrelationLengthEnvelope(self, aCorrFunction: callable) -> Tuple[float, float, float]:
        '''
        The autocorrelation parameter has to be in the second position of the interpolating function
        '''
        upperPeaks, lowerPeaks = self.envelopePeaks()
        interFuncUpper = self.getInterpolatingFunction(upperPeaks)
        interFuncLower = self.getInterpolatingFunction(lowerPeaks)
        poptUpper, _, _ = self.getAutocorrelationFit(aCorrFunction, x=self.xAxis,
                                                     y=self.applyInterpolatingFunction(interFuncUpper, self.xAxis))
        poptLower, _, _ = self.getAutocorrelationFit(aCorrFunction, x=self.xAxis,
                                                     y=self.applyInterpolatingFunction(interFuncLower, self.xAxis))

        return head2(poptUpper), head2(poptLower), (head2(poptUpper) + head2(poptLower))/2

    def getCorrelationParameters(self, aCorrFunction: callable) -> np.array:
        '''
        The autocorrelation parameter has to be in the second position of the interpolating function
        '''

        popt, _, _ = self.getAutocorrelationFit(aCorrFunction, x=self.lags,
                                                y=self.autocorrelation)

        return popt

    @classmethod
    def instantiateFromNpy(cls, npyPath) -> None:

        matrix = np.load(npyPath)

        for i in range(matrix.shape[0]):
            autoCorrelation(matrix[i, :], idNumber=i + 1)

    @classmethod
    def instantiateFromNpyTS(cls, npyPath, log: bool = True) -> None:

        matrix = np.load(npyPath)
        nlags = matrix.shape[1] - 1
        for i in range(matrix.shape[0]):
            if log:
                autocorrelation: np.ndarray = acf(np.exp(matrix[i, :]), fft=True,
                                                  nlags=nlags)
            else:
                autocorrelation: np.ndarray = acf(matrix[i, :], fft=True,
                                                  nlags=nlags)
            autoCorrelation(autocorrelation, idNumber=i + 1)

    @classmethod
    def instantiateFromCsvTS(cls, csvPath, log: bool = True, limit: Optional[int] = None) -> None:

        matrix: np.ndarray = np.loadtxt(csvPath, delimiter=',')
        if limit is not None:
            assert limit >= 0, "Limit should be positive"
            matrix: np.ndarray = matrix[:, -limit:]
        nlags = matrix.shape[1] - 1
        for i in range(matrix.shape[0]):
            if log:
                autocorrelation: np.ndarray = acf(np.exp(matrix[i, :]), fft=True,
                                                  nlags=nlags)
            else:
                autocorrelation: np.ndarray = acf(matrix[i, :], fft=True,
                                                  nlags=nlags)
            autoCorrelation(autocorrelation, idNumber=i + 1)
