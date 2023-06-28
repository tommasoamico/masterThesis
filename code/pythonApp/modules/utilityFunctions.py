import numpy as np
from typing import Iterable, Union, Tuple, List
from infix import shift_infix as infix
from scipy.signal import find_peaks
from scipy.stats import iqr
from modules.constantsAndVectors import findPeaksParams, blues
import pandas as pd
from functools import reduce
from matplotlib.colors import Colormap
import matplotlib.pyplot as plt
from scipy.stats import linregress
from typing import Tuple, Generator, Dict
from typeguard import typechecked
from pathlib import Path


@infix
def fC(f: callable, g: callable) -> callable:
    return lambda x: f(g(x))


def autocorrelationProduct(x, eta, xi):
    # return 1 + x**(-eta) * np.exp(-x/xi)
    return x**(-eta) * np.exp(-x/xi)


def autocorrelationExponential(x, a, xi):
    return a + np.exp(-x/xi)


def rangeAutocorrelation(autocorrelation: Iterable) -> np.ndarray:
    # return np.array([1e-8] + list(range(1, len(autocorrelation))))
    return np.arange(1, len(autocorrelation) + 1)


def getPeaks(array: np.array) -> np.ndarray:
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


def cdfLogSpacePositiveH(x: Union[float, np.ndarray], xb: Union[float, np.ndarray], gamma: Union[float, np.ndarray], h: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    return 1 - np.exp((-2*np.exp(x) + np.exp(xb))*gamma) * ((2*np.exp(x) + h)**((-1 + h) * gamma))*((np.exp(xb) + h)**(gamma - h*gamma))


def plotMomentScaling(sizesPath: Path, maxK: int = 3, title: str = '$<m^k>$ vs $<m>$', suffix: str = '') -> None:
    assert maxK <= 5, "Maximum k available is 5"
    assert isinstance(maxK, int), 'available data is in int format'
    # colorsMoment: List[Tuple[float]] = getColorsFromColormap(
    #   blues, maxK - 1, initialPoint=.3)
    allMeans = np.load(sizesPath / f'allMeans{suffix}.npy')
    allMoments = np.load(sizesPath / f'allMoments{suffix}.npy')

    for i, k in enumerate(range(2, maxK + 1)):
        plt.scatter(allMeans[i], allMoments[i],
                    label=f'k={k}', edgecolor='black')

    plt.xlabel('$<m>$', fontsize=15)
    plt.ylabel(f'$<m^k>$', fontsize=15)
    plt.title(title, fontsize=18)
    plt.legend(facecolor='aliceblue', shadow=True, edgecolor='black')


@typechecked
def fitMoments(sizesPath: Path, maxK: int = 3, suffix: str = '') -> Dict[str, Dict[str, float]]:
    assert maxK <= 5, "Maximum k available is 5"
    assert isinstance(maxK, int), 'available data is in int format'
    allMeans: np.ndarray = np.load(sizesPath / f'allMeans{suffix}.npy')
    allMoments: np.ndarray = np.load(sizesPath / f'allMoments{suffix}.npy')
    dictResult: Dict[str, dict] = {'linear': {}, 'log': {}}
    for i, _ in enumerate(range(2, maxK + 1)):
        fitResult = linregress(allMeans[i, :], allMoments[i, :])
        dictResult['linear'][f'k={i + 2}'] = fitResult.slope
        fitResultLog = linregress(
            np.log10(allMeans[i, :]), np.log10(allMoments[i, :]))
        dictResult['log'][f'k={i + 2}'] = fitResultLog.slope
    return dictResult


@typechecked
def plotCorrelationMeanSize(path: Path, suffix: str = '') -> None:
    parametersPath: Path = path / f'correlationParameters{suffix}.csv'
    correlationLenghts: pd.Series = pd.read_csv(parametersPath)['correlation']
    allMeans: np.ndarray = np.load(
        path / 'momentScaling' / f'allMeans{suffix}.npy')

    plt.scatter(allMeans[0, :], correlationLenghts, edgecolor='black')
    plt.xlabel('$<m>$', fontsize=15)
    plt.ylabel('$\\xi$', fontsize=15)
    _ = plt.title('autocorrelation length vs mean size at birth', fontsize=18)


def susmanProcessing(pathCsv: str):
    allArrays = []

    df: pd.DataFrame = pd.read_csv(pathCsv)
    df['lineage_ID'] = df['lineage_ID'].astype(int)
    uniqueLineages: np.ndarray = df['lineage_ID'].unique()

    for lineage in uniqueLineages:
        newDf: pd.DataFrame = df[df['lineage_ID'] == lineage]
        tmp = np.array(newDf['length_birth'])

        if len(tmp) < 228:
            pass
        else:
            allArrays.append(tmp[:228])
    finalArray = reduce(lambda x, y: np.vstack([x, y]), allArrays)

    return finalArray.T


def tanouchiProcessing(pathCsv, seq_len: int):
    df: pd.DataFrame = pd.read_csv(pathCsv)

    df['lineage_ID'] = df['lineage_ID'].astype(int)
    uniqueLineages: np.ndarray = df['lineage_ID'].unique()
    nLineages: int = int(uniqueLineages[-1])
    finalArray: np.ndarray = np.zeros((69, nLineages))
    for lineage in uniqueLineages:
        newDf: pd.DataFrame = df[df['lineage_ID'] == lineage]
        finalArray[:, lineage-1] = np.array(newDf['length_birth'])[:69]
    return finalArray



