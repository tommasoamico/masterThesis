from typing import List, Tuple
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

gammaGammaC: np.array = np.logspace(-1, -3, 100)
gammaValues: np.array = criticalPoint - gammaGammaC

##################################################################
# Subject to change depending on the simulation, h > 0 case      #
##################################################################

# []  # np.linspace(.75, .95, 100)
# np.linspace(.32, .425, 100)
hValueses: np.ndarray = np.linspace(.75, .95, 100)
gammaPositiveH: float = 15  # criticalPoint + 1
leftBracketPositiveH = -400
rightBracketPositiveH = 10


slopeSimulation2: float = 1.9496974329535792
interceptSimulation2: float = 0.09660761792804673
slopeSimulation3: float = 2.854356191797568
interceptSimulation3: float = 0.2781725594651162


arimaParameters: List[Tuple[int, int, int]] = [
    (x, y, z) for x in range(4) for y in range(4) for z in range(4)]
