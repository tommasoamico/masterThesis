from modules.utilityFunctions import inverseCum, brackets, cdfLogSpacePositiveH, cdfLogSpace
import numpy as np
from scipy.optimize import root_scalar
from typing import Tuple, Iterable, Type
from statsmodels.tsa.stattools import acf
from modules.constantsAndVectors import leftBracketPositiveH, rightBracketPositiveH


class simulation:
    all = []

    def __init__(self, gamma: float, seriesLength: float, h: float = 0.) -> None:
        self.gamma: float = gamma
        self.seriesLength: float = seriesLength
        self.h = h

        simulation.all.append(self)

    def __len__(self) -> int:
        return self.seriesLength

    def __enter__(self) -> None:
        return self

    def __exit__(self) -> None:
        print('Exiting...')

    def __repr__(self) -> str:
        return f'{self.__class__.__name__} instance; parameters =  {self.__dict__}'

    def drawCDF(self, unif: float, xb: float) -> float:
        if xb > -20:
            leftBracket, rightBracket = brackets(xb)
            assert (np.sign(cdfLogSpace(leftBracket, gamma=self.gamma, xb=xb) - unif) *
                    np.sign(cdfLogSpace(rightBracket, gamma=self.gamma, xb=xb) - unif)) == -1, "CDF is not bracketed"

            return root_scalar(lambda t: cdfLogSpace(t, gamma=self.gamma, xb=xb) - unif, bracket=[leftBracket, rightBracket], method='brentq').root
        else:
            return - np.log(2) - (1 / self.gamma) * np.log(np.random.uniform(size=1)) + xb

    def drawCdfPositiveH(self, unif: float, xb: float):
        root_scalar(lambda t: cdfLogSpacePositiveH(t, gamma=self.gamma, xb=xb, h=self.h) -
                    unif, bracket=[leftBracketPositiveH, rightBracketPositiveH], method='brentq').root

    def sizeAtBirth(self, xb0: float) -> np.array:
        uniformDraws = np.random.uniform(size=self.seriesLength)

        logSizes = np.zeros(self.seriesLength)
        logSizes[0] = xb0
        for i in range(1, self.seriesLength):
            logSizes[i] = self.drawCDF(unif=uniformDraws[i], xb=logSizes[i-1])

        return logSizes

    @staticmethod
    def drawUniform() -> float:
        return np.random.uniform(size=1)

    def startingPoint(self, b, delta) -> float:
        return inverseCum(u=self.drawUniform(), b=b, delta=delta)

    def simulate(self, b: float = 10, delta: float = .1) -> Tuple[np.array]:
        logSizes = self.sizeAtBirth(xb0=np.log(
            self.startingPoint(b=b, delta=delta)))
        autocorrelation = acf(np.exp(logSizes), fft=True,
                              nlags=len(logSizes) - 1)

        return logSizes, autocorrelation

    def simulatePositiveH(self, b: float = 10, delta: float = .1) -> Tuple[np.array]:
        pass

    @classmethod
    def instantiateFromIterable(cls, gammaValues: Iterable, seriesLength: int) -> None:
        for gamma in gammaValues:
            simulation(gamma=gamma, seriesLength=seriesLength)
