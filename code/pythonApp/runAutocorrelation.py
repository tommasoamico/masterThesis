import pandas as pd
import numpy as np
from modules.autocorrelation import autoCorrelation
from statsmodels.tsa.stattools import acf
import matplotlib.pyplot as plt
from modules.utilityFunctions import autocorrelationExponential, autocorrelationProduct
from modules.autocorrelation import autoCorrelation
from typing import Type, Tuple
from tqdm import tqdm

autoCorrelation.all = []
# tanouchiPath = '/Users/tommaso/Desktop/masterThesis/data/realData/Tanouchi/Tanouchi25C.csv'
acorrPath = '/Users/tommaso/Desktop/masterThesis/data/positiveH/absorbingHDeltaDecidingShort/autoCorrelations.npy'

saveResultsPath = '/Users/tommaso/Desktop/masterThesis/data/positiveH/absorbingHDeltaDecidingShort/correlationParameters.csv'

autoCorrelation.instantiateFromNpy(acorrPath)

with open(saveResultsPath, mode='w') as f:
    f.write('exponent,correlation,id')
    for instance in tqdm(autoCorrelation.all):
        if np.any(np.isnan(instance.autocorrelation)):
            popt = (np.nan, np.nan)
        else:
            popt = instance.getCorrelationParameters(autocorrelationProduct)
        f.write(f'\n{popt[0]},{popt[1]},{instance.idNumber}')
