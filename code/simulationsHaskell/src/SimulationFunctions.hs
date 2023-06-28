{-# LANGUAGE ImportQualifiedPost #-}

module SimulationFunctions
  (simulate, simulateNotMonadic,)
where


import Data.Vector qualified as DV
import DataTypes (CdfParameters (..), LogSize, SimulationParameters (..), SimulationParametersNotMonadic (..))
import DrawAlgorithm (drawCDF)
import Control.Monad.State
    ( MonadTrans(lift),
      StateT,
      gets,
      MonadState(put, get),
      execStateT )
import System.Random (randomR)
import Numeric.RootFinding (Root, fromRoot)


-- sizeAtBirth :: series length -> StateT state Monad resultComputation
sizeAtBirth :: Int -> StateT SimulationParameters Root LogSize
sizeAtBirth 0 = gets (xBirth . params)
sizeAtBirth n = do
  paramDraw <- get
  let cdfParams = params paramDraw
  let currentTs = accumulatedDraw paramDraw
  let gen = generator paramDraw
  let (unif, gen') = randomR (0,1) gen
  newDraw <- lift $ drawCDF cdfParams unif
  let newTs = DV.snoc currentTs newDraw
  put $ paramDraw {generator = gen', accumulatedDraw = newTs, params = cdfParams{xBirth = newDraw}}
  sizeAtBirth (n-1)

-- sizeAtBirthNotMonadic :: seriesLength -> Initial Vector -> initialOarameters -> Result
sizeAtBirthNotMonadic :: Int -> DV.Vector LogSize -> SimulationParametersNotMonadic -> DV.Vector LogSize
sizeAtBirthNotMonadic 0 currentTS _ = currentTS
sizeAtBirthNotMonadic n currentTS parameters = sizeAtBirthNotMonadic (n-1) newTs newParameters
  where
    cdfParameters = paramsNotMonadic parameters
    (unif, gen') = randomR (0,1) (generatorNotMonadic parameters)
    newDraw = drawCDF cdfParameters unif
    newTs = DV.snoc currentTS $ fromRoot 0 newDraw
    newParameters = SimulationParametersNotMonadic {generatorNotMonadic = gen', paramsNotMonadic = cdfParameters{xBirth = fromRoot 0 newDraw}}




simulate :: Int -> SimulationParameters -> Root (DV.Vector Double)
simulate n paramDraw =  accumulatedDraw <$> execStateT (sizeAtBirth n) paramDraw

simulateNotMonadic :: Int -> SimulationParametersNotMonadic -> DV.Vector Double
simulateNotMonadic n paramDraw = sizeAtBirthNotMonadic n DV.empty paramDraw

{-
simulate :: Int -> SimulationParameters -> Root SimulationParameters
simulate n paramDraw =  execStateT (sizeAtBirth n) paramDraw
-}
