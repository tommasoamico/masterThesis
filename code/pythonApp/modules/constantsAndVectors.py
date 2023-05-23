from typing import List
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Colormap

hieghtFindMaxPeaks: List[int] = [0, 1]  # Closed interval
distancePeaksDefault: int = 5
findPeaksParams: dict = {'height': hieghtFindMaxPeaks,
                         'distance': distancePeaksDefault}
criticalPoint: float = 1 / np.log(2)
blues: Colormap = plt.get_cmap('Blues')

##################################################################
# Subject to change depending on the simulation, h = 0 case      #
##################################################################

gammaGammaC: np.array = np.logspace(np.log10(1), np.log10(.1), 100)
gammaValues: np.array = criticalPoint - gammaGammaC

##################################################################
# Subject to change depending on the simulation, h > 0 case      #
##################################################################

hValueses: np.array = np.logspace(
    np.log10(1e-3), np.log10(2 * criticalPoint), 100)
gammaPositiveH: float = 2 * criticalPoint
leftBracketPositiveH = -400
rightBracketPositiveH = 10
