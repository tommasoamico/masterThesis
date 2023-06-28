import pandas as pd
import numpy as np
from modules.arima import arimaHandler
from modules.constantsAndVectors import arimaParameters
from typing import List, Tuple
from tqdm import tqdm
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from pprint import pprint
from statsmodels.graphics.tsaplots import plot_predict

tanouchi25Path = '/Users/tommaso/Desktop/masterThesis/data/realData/Tanouchi/Tanouchi25/Tanouchi25C.csv'


def instantiateFromCsv(pathCsv: str) -> np.ndarray:
    df: pd.DataFrame = pd.read_csv(pathCsv)
    df['lineage_ID'] = df['lineage_ID'].astype(int)
    uniqueLineages: np.array = df['lineage_ID'].unique()
    for lineage in uniqueLineages:
        newDf: pd.DataFrame = df[df['lineage_ID'] == lineage]
    return np.array(newDf['length_birth'])


with arimaHandler(instantiateFromCsv(tanouchi25Path)) as arima:
    finalPrediction = arima.getPrediciton(order=(1, 0, 1), lenPrediction=5000)
    plt.plot(finalPrediction)
    plt.show()

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
