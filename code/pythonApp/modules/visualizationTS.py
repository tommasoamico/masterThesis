import numpy as np
import pandas as pd
from typing import List, Tuple, Union, Optional
from tqdm import tqdm
from typeguard import typechecked
from sklearn.metrics import mean_squared_error as mse
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes


class systemHandling:
    allInstances: List["systemHandling"] = []

    def __init__(self, allMeans: np.ndarray, allMoments: np.ndarray) -> None:
        self.allMeans = allMeans
        self.allMoments = allMoments
        self.allInstances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        print('Exiting...')

    def __repr__(self) -> str:
        return f"systemInstance firstMean={self.allMeans[0,0]}"

    @typechecked
    def getHeatmapGrid(self, slopeRange: np.ndarray, interceptRange: np.ndarray, kIdx: int = 0) -> np.ndarray:
        grid: np.ndarray = np.zeros((len(slopeRange), len(interceptRange)))
        for i in tqdm(range(len(slopeRange))):
            for j in range(len(interceptRange)):
                yPred = np.log10(
                    self.allMeans[0, :]) * slopeRange[i] + interceptRange[j]
                error = mse(y_true=np.log10(
                    self.allMoments[kIdx, :]), y_pred=yPred)
                grid[i, j] = error
        return grid

    @typechecked
    def lrLandscape(self, slopeRange: np.ndarray, interceptRange: np.ndarray, scatterPoints: Optional[List[Tuple[float, float, str]]] = None, savePath: Optional[str] = None, show: bool = False, title: Optional[str] = None, kIdx: int = 0) -> None:
        assert len(slopeRange) == len(interceptRange), "We want a square grid"
        grid: np.ndarray = self.getHeatmapGrid(
            slopeRange=slopeRange, interceptRange=interceptRange, kIdx=kIdx)
        fig, ax = plt.subplots()

        tickSeparation: int = len(slopeRange) // 10
        im = ax.imshow(grid, cmap='hot', interpolation='nearest')
        ax.set_xticks(range(len(slopeRange))[::tickSeparation], labels=list(
            map(lambda x: round(x, 2), slopeRange[::tickSeparation])))
        ax.set_yticks(range(len(interceptRange))[::tickSeparation], list(
            map(lambda x: round(x, 2), interceptRange[::tickSeparation])))
        fig.colorbar(im)

        if scatterPoints is not None:
            for scatter in scatterPoints:
                xPoint = (scatter[0] - np.min(slopeRange)) * \
                    (len(slopeRange) / (np.max(slopeRange) - np.min(slopeRange)))
                yPoint = (scatter[1] - np.min(interceptRange)) * (
                    len(interceptRange) / (np.max(interceptRange) - np.min(interceptRange)))
                plt.scatter(xPoint, yPoint, label=scatter[2])
            plt.legend()
        if savePath is not None:
            fig.savefig(savePath, dpi=300)

        if title is not None:
            plt.title(title)
        if show:
            plt.show()

    @typechecked
    def lr3d(self, slopeRange: np.ndarray, interceptRange: np.ndarray, scatterPoints: Optional[List[Tuple[float, float, str]]] = None, savePath: Optional[str] = None, show: bool = False, title: Optional[str] = None, kIdx: int = 0) -> None:
        grid: np.ndarray = self.getHeatmapGrid(
            slopeRange=slopeRange, interceptRange=interceptRange, kIdx=kIdx)
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        from matplotlib import cm
        X, Y = np.meshgrid(slopeRange, interceptRange)
        ax.plot_surface(X, Y, grid, cmap=cm.coolwarm, linewidth=0)
        plt.show()

    @classmethod
    def instantiateFromList(cls, listMeans: List[np.ndarray], listMoments: List[np.ndarray]) -> None:
        for means, moments in zip(listMeans, listMoments):
            cls(allMeans=means, allMoments=moments)
