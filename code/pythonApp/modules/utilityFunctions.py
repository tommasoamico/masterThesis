import numpy as np
from typing import Iterable, Union, Tuple, List
from infix import shift_infix as infix
from scipy.signal import find_peaks
from scipy.stats import iqr
from modules.constantsAndVectors import findPeaksParams
import pandas as pd
from functools import reduce
from matplotlib.colors import Colormap


@infix
def fC(f: callable, g: callable) -> callable:
    return lambda x: f(g(x))


def autocorrelationProduct(x, eta, xi):
    # return 1 + x**(-eta) * np.exp(-x/xi)
    return x**(-eta) * np.exp(-x/xi)


def autocorrelationExponential(x, a, xi):
    return a + np.exp(-x/xi)


def rangeAutocorrelation(autocorrelation: Iterable) -> np.array:
    # return np.array([1e-8] + list(range(1, len(autocorrelation))))
    return np.arange(1, len(autocorrelation) + 1)


def getPeaks(array: np.array) -> np.array:
    peaks, _ = find_peaks(
        array, **findPeaksParams)

    peaks = np.concatenate([np.zeros(1, dtype=int), peaks, np.ones(
        1, dtype=int) * (len(array) - 1)])

    return peaks


def tail(element: Iterable) -> Iterable:
    return element[1:]


def head2(element: Iterable) -> Iterable:
    return element[1]

#################
# h = 0 case    #
#################


def cdfLogSpace(x: Union[float, np.array], xb: Union[float, np.array], gamma: Union[float, np.array]) -> Union[float, np.array]:
    return 1 - 2**(- gamma) * np.exp(gamma * (-2 * np.exp(x) + np.exp(xb) - x + xb))


def pdfLogSpace(x: Union[float, np.array], xb: Union[float, np.array], gamma: Union[float, np.array]) -> Union[float, np.array]:
    return 2**(- gamma) * gamma * np.exp(gamma*(- x + xb + np.exp(xb) - 2*np.exp(x)) + np.log(1 + 2 * np.exp(x)))


def cumulative(b: Union[float, np.array], delta: Union[float, np.array], m: Union[float, np.array]) -> Union[float, np.array]:
    return b**(-delta) * m**(delta)


#########################
# End of the h = 0 case #
#########################


def cumulative_data(array):
    array = np.array(array)
    array = np.sort(array)
    array = array[~np.isnan(array)]
    array = array[array >= 0]
    cumul = 1 - np.arange(0, len(array))/(len(array))
    return array, cumul


def lineages(df: pd.DataFrame, columnID: str = 'lineage_ID', lengthColumn: str = 'length_birth') -> List[np.array]:
    uniqueLineages: np.array = df[columnID].unique()
    returnList = []
    for lineage in uniqueLineages:
        newDf: pd.DataFrame = df[df[columnID] == lineage]
        returnList.append(np.array(newDf[lengthColumn]))

    return returnList


def stackList(inputList: List[np.array]) -> np.array:
    return reduce(lambda x, y: np.vstack([x, y]), inputList)


def freedmanDiaconis(data: np.array) -> int:
    IQR = iqr(data, rng=(25, 75), scale=1)
    n = len(data)
    bw = (2 * IQR) / np.power(n, 1/3)

    datMin, datMax = np.min(data), np.max(data)
    dataRange = datMax - datMin
    fdBins = int((dataRange / bw) + 1)

    return fdBins


def getColorsFromColormap(cmap: Colormap, nColors: int, initialPoint: float = 0., endPoint: float = 1.) -> List[Tuple[float]]:
    return [cmap(i) for i in np.linspace(initialPoint, endPoint, nColors)]


def inverseCum(b: Union[float, np.array], delta: Union[float, np.array], u: Union[float, np.array]) -> Union[float, np.array]:
    return b * u**(1/delta)


def brackets(xb: float) -> Tuple[float]:
    if xb < -7:
        leftBracket = -np.log(2*(1 - 0.5)) + xb + np.exp(xb) - 2
    else:
        leftBracket = -25

    rightBracket = 10

    return leftBracket, rightBracket


def powerLaw(x, a, b):
    return a*x**b


def cdfLogSpacePositiveH(x: Union[float, np.array], xb: Union[float, np.array], gamma: Union[float, np.array], h: Union[float, np.array]) -> Union[float, np.array]:
    return 1 - np.exp((-2*np.exp(x) + np.exp(xb))*gamma) * ((2*np.exp(x) + h)**((-1 + h) * gamma))*((np.exp(xb) + h)**(gamma - h*gamma))
