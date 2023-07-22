module ConstantsAndVectors
  ( criticalPoint,
    bracketPositiveH,
    gammaSim,
    hSim,
    hSimList,
    simulationParams,
    lengthSimulation,
    simulationParamsNotMonadic,
    savePath
  )
where


import DataTypes (Bracket, CdfParameters (..), SimulationParameters (..), SimulationParametersNotMonadic(..))
import UtilityFunctions (startingPointLog)
import System.Random (StdGen)
import Data.Sequence (empty)
import DSP.Basic (linspace)



lengthSimulation :: Int
lengthSimulation = 400000

criticalPoint :: Double
criticalPoint = 1 / log 2

gammaSim :: Double
gammaSim = criticalPoint + 0.01

hSim::Double
hSim = 8

hSimList :: [Double]
hSimList = linspace 8 13 100

bracketPositiveH :: Bracket
bracketPositiveH = (-350, 10)


simulationParamsCdf :: StdGen -> CdfParameters
simulationParamsCdf gen = CdfParameters {xBirth = startingPointLog hSim gen, gammaValue = gammaSim, hValue = hSim}

simulationParams :: StdGen -> SimulationParameters
simulationParams gen = SimulationParameters {params = simulationParamsCdf gen, generator = gen, accumulatedDraw = empty}

simulationParamsNotMonadic :: StdGen -> SimulationParametersNotMonadic
simulationParamsNotMonadic gen = SimulationParametersNotMonadic {paramsNotMonadic = simulationParamsCdf gen, generatorNotMonadic = gen}

savePath :: FilePath
savePath = "../../data/simulationHaskell/simulation2/simulation2.csv"

{-
myTolerance :: Tolerance
myTolerance = AbsTol 1e-3

myRiddersParam :: RiddersParam
myRiddersParam = RiddersParam {riddersTol = myTolerance, riddersMaxIter = 10000}
-}