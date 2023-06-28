from modules.utilityFunctions import inverseCum, brackets, cdfLogSpacePositiveH, cdfLogSpace
import numpy as np
from scipy.optimize import root_scalar
from typing import Tuple, Iterable, Type
from statsmodels.tsa.stattools import acf
from modules.constantsAndVectors import leftBracketPositiveH, rightBracketPositiveH


class simulation:
    all = []

    def __init__(self, gamma: float, seriesLength: float, h: float = 0) -> None:
        self.gamma: float = gamma

        self.seriesLength: float = seriesLength
        self.h = h
        self.hSign = np.sign(h)
        self.omega2 = self.gamma
        self.u = self.h
        simulation.all.append(self)

    def __len__(self) -> int:
        return self.seriesLength

    def __enter__(self) -> None:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
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

            return - np.log(2) - (1 / self.gamma) * np.log(unif) + xb

    def drawCdfPositiveH(self, unif: float, xb: float) -> float:
        return root_scalar(lambda t: cdfLogSpacePositiveH(t, gamma=self.gamma, xb=xb, h=self.h) -
                           unif, bracket=[leftBracketPositiveH, rightBracketPositiveH], method='brentq').root

    def sizeAtBirth(self, xb0: float) -> np.ndarray:

        if self.hSign == 0:
            drawFunc = self.drawCDF
        else:
            drawFunc = self.drawCdfPositiveH
        uniformDraws = np.random.uniform(size=self.seriesLength)

        logSizes = np.zeros(self.seriesLength)
        logSizes[0] = xb0
        for i in range(1, self.seriesLength):
            logSizes[i] = drawFunc(unif=uniformDraws[i],
                                   xb=logSizes[i-1])

        return logSizes

    @staticmethod
    def drawUniform() -> float:
        return np.random.uniform(size=1)

    def startingPoint(self, b: float, delta: float) -> float:

        if self.hSign == 0:
            return inverseCum(u=self.drawUniform(), b=b, delta=delta)
        else:
            return self.h

    def simulate(self, b: float = 10, delta: float = .1) -> Tuple[np.ndarray]:
        logSizes = self.sizeAtBirth(xb0=np.log(
            self.startingPoint(b=b, delta=delta)))
        autocorrelation = acf(np.exp(logSizes), fft=True,
                              nlags=len(logSizes) - 1)

        return logSizes, autocorrelation

    @classmethod
    def instantiateFromIterableGammas(cls, gammaValues: Iterable, seriesLength: int, h: float) -> None:
        for gamma in gammaValues:
            simulation(gamma=gamma, seriesLength=seriesLength, h=h)

    @classmethod
    def instantiateFromIterableHs(cls, gamma: float, seriesLength: int, hValues: Iterable) -> None:
        for h in hValues:
            simulation(gamma=gamma, seriesLength=seriesLength, h=h)

    def __cdfProtein(self, mb: float, t: float, t0: float):
        if t > t0:
            # * (self.h), (1-mb)/(self.u+1))
            surv = - ((mb/(self.u+1)) * (self.gamma) * (np.exp(t)-np.exp(t0)) +
                      ((1-mb)/(self.u+1)) * self.omega2 * (t-t0))
        else:
            surv = 0

        return np.exp(surv)

    def __drawTauNumerical(self, unif: float, mb: float) -> float:
        t0: float = np.max([0,  np.log(1 + (self.u/mb))])

        tau: float = root_scalar(lambda t: self.__cdfProtein(mb=mb, t=t, t0=t0) - unif,
                                 bracket=[t0, 40], method='brentq').root

        return tau

    @staticmethod
    def __mFunction(time: float, mb: float) -> float:
        return mb * np.exp(time)

    def __sizeAtBirthProtein(self, mb0: float) -> np.ndarray:
        assert self.hSign > 0, "h should be greater than 0"

        uniformDraws: np.ndarray = np.random.uniform(size=self.seriesLength)
        sizes: np.ndarray = np.zeros(self.seriesLength)
        sizes[0]: float = mb0

        for i in range(1, self.seriesLength):
            divisionTime = self.__drawTauNumerical(
                uniformDraws[i], mb=sizes[i-1]),

            sizes[i] = self.__mFunction(time=divisionTime, mb=sizes[i - 1]) / 2

        return sizes

    def simulateProtein(self) -> Tuple[np.ndarray]:
        sizes: np.ndarray = self.__sizeAtBirthProtein(
            mb0=self.startingPoint(b=10, delta=.1))
        autocorrelation: np.ndarray = acf(sizes, fft=True,
                                          nlags=len(sizes) - 1)

        return sizes, autocorrelation
