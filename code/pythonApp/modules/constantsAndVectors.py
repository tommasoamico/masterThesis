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

gammaGammaC: np.array = [1]  # np.logspace(-1, -3, 100)
gammaValues: np.array = criticalPoint - gammaGammaC

##################################################################
# Subject to change depending on the simulation, h > 0 case      #
##################################################################

hValueses: np.array = np.linspace(1e-1, 1e-3, 100)
gammaPositiveH: float = criticalPoint - .2
leftBracketPositiveH = -400
rightBracketPositiveH = 10
