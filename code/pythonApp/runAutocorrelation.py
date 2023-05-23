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
longLineagesPath = '/Users/tommaso/Desktop/masterThesis/data/nullH1MilionSamples/autoCorrelations.npy'

saveResultsPath = '/Users/tommaso/Desktop/masterThesis/data/nullH1MilionSamples/correlationParameters.csv'

autoCorrelation.instantiateFromNpy(longLineagesPath)

with open(saveResultsPath, mode='w') as f:
    f.write('exponent,correlation,id')
    for instance in tqdm(autoCorrelation.all):
        popt = instance.getCorrelationParameters(autocorrelationProduct)
        f.write(f'\n{popt[0]},{popt[1]},{instance.idNumber}')
