from modules.markovChainModule import markovChain
from modules.utilityFunctions import stackList, freedmanDiaconis
import numpy as np
from functools import reduce
import matplotlib.pyplot as plt
from typing import List, Type, Tuple
from pathlib import Path
'''
pathCsv: str = '/Users/tommaso/Desktop/masterThesis/data/realData/Tanouchi/Tanouchi37/Tanouchi37C.csv'

markovChain.all: List[Type[markovChain]] = []


markovChain.instantiateFromCsv(pathCsv=pathCsv)


allLengths: List[int] = [len(instance) for instance in markovChain.all]


minLength: int = np.min(allLengths)

allAutocorrelations: List[np.array] = [instance.autocorrelation(
    maxLags=minLength) for instance in markovChain.all]


averagedAutocorrelation: np.array = reduce(
    lambda x, y: x + y, allAutocorrelations, np.zeros(minLength)) / len(allAutocorrelations)

savePathAutocorrelation: str = '/Users/tommaso/Desktop/masterThesis/data/realData/Susman/longLineages/averagedAutocorrelation.npy'

# np.save(savePathAutocorrelation, averagedAutocorrelation)

################
# PDF          #
################

allPdfs: List[np.array] = [instance.pdf(
    bins='fd') for instance in markovChain.all]


allCounts: List[np.array] = list(map(lambda x: x[0], allPdfs))

allCumulativeData: List[Tuple[np.array]] = [
    instance.survival_data() for instance in markovChain.all]

allBinCenters: List[np.array] = list(map(lambda x: x[1], allPdfs))

allMeans: List[float] = [instance.meanSizeAtBirth()
                         for instance in markovChain.all]

xAxises: List[np.array] = [x/y for x, y in zip(allBinCenters, allMeans)]

maxXLength: int = np.max([len(x) for x in xAxises])

yAxises: List[np.array] = [x * y for x, y in zip(allCounts, allBinCenters)]

maxYLength: int = np.max([len(y) for y in xAxises])

################
# Survival     #
################

allCumulativesXData: List[np.array] = list(
    map(lambda x: x[0], allCumulativeData))

allCumulativesYData: List[np.array] = list(
    map(lambda x: x[1], allCumulativeData))

xAxisesCumulative: List[np.array] = [
    x/y for x, y in zip(allCumulativesXData, allMeans)]

maxXCumulative = np.max([len(x) for x in xAxisesCumulative])

yAxisesCumulative: List[np.array] = [x * y for x,
                                     y in zip(allCumulativesYData, allCumulativesXData)]

maxYCumulative = np.max([len(y) for y in yAxisesCumulative])


###############
# Saving      #
###############

savePdfCounts: np.array = stackList(list(map(lambda x: np.pad(x[0], pad_width=(
    0, maxXLength - len(x[0])), mode='constant', constant_values=(np.nan,)), allPdfs)))

savePdfCenters: np.array = stackList(list(map(lambda x: np.pad(x[1], pad_width=(
    0, maxXLength - len(x[1])), mode='constant', constant_values=(np.nan,)), allPdfs)))

saveXMatrixCollapse: np.array = stackList(list(map(lambda x: np.pad(x, pad_width=(
    0, maxXLength - len(x)), mode='constant', constant_values=(np.nan,)), xAxises)))

saveYMatrixCollapse: np.array = stackList(list(map(lambda y: np.pad(y, pad_width=(
    0, maxYLength - len(y)), mode='constant', constant_values=(np.nan,)), yAxises)))

saveSurvivalX: np.array = stackList(list(map(lambda x: np.pad(x, pad_width=(
    0, maxXCumulative - len(x)), mode='constant', constant_values=(np.nan,)), allCumulativesXData)))

saveSurvivalY: np.array = stackList(list(map(lambda x: np.pad(x, pad_width=(
    0, maxYCumulative - len(x)), mode='constant', constant_values=(np.nan,)), allCumulativesYData)))

saveXMatrixSurvival: np.array = stackList(list(map(lambda x: np.pad(x, pad_width=(
    0, maxXCumulative - len(x)), mode='constant', constant_values=(np.nan,)), xAxisesCumulative)))

saveYMatrixSurvival: np.array = stackList(list(map(lambda y: np.pad(y, pad_width=(
    0, maxYCumulative - len(y)), mode='constant', constant_values=(np.nan,)), yAxisesCumulative)))

savePathXAxisCollapsePdf: str = '/Users/tommaso/Desktop/masterThesis/data/realData/Tanouchi/longLineages/xAxisesCollapsePdf.npy'
savePathYAxisCollapsePdf: str = '/Users/tommaso/Desktop/masterThesis/data/realData/Tanouchi/longLineages/yAxisesCollapsePdf.npy'

savePathPdfCounts: str = '/Users/tommaso/Desktop/masterThesis/data/realData/Tanouchi/longLineages/pdfCounts.npy'
savePathPdfCenters: str = '/Users/tommaso/Desktop/masterThesis/data/realData/Tanouchi/longLineages/pdfCenters.npy'

savePathXAxisCollapseSurvival: str = '/Users/tommaso/Desktop/masterThesis/data/realData/Tanouchi/longLineages/xAxisesCollapseSurvival.npy'
savePathYAxisCollapseSurvival: str = '/Users/tommaso/Desktop/masterThesis/data/realData/Tanouchi/longLineages/yAxisesCollapseSurvival.npy'

savePathXAxisSurvival: str = '/Users/tommaso/Desktop/masterThesis/data/realData/Tanouchi/Tanouchi37/xAxisesSurvival.npy'
savePathYAxisSurvival: str = '/Users/tommaso/Desktop/masterThesis/data/realData/Tanouchi/Tanouchi37/yAxisesSurvival.npy'


# np.save(savePathXAxisCollapsePdf, saveXMatrixCollapse)
# np.save(savePathYAxisCollapsePdf, saveYMatrixCollapse)

# np.save(savePathPdfCounts, savePdfCounts)
# np.save(savePathPdfCenters, savePdfCenters)

# np.save(savePathXAxisCollapseSurvival, saveXMatrixSurvival)
# np.save(savePathYAxisCollapseSurvival, saveYMatrixSurvival)

# np.save(savePathXAxisSurvival, saveSurvivalX)
# np.save(savePathYAxisSurvival, saveSurvivalY)
'''
'''
dataPath: Path = Path.cwd().parents[1] / 'data'
susmanData: Path = dataPath / 'realData' / \
    'Susman' / 'longLineages' / 'susmanDataLL.csv'
'''
