{-# LANGUAGE ImportQualifiedPost #-}

module SimulationFunctions
  ( simulationMatrix,
    sizeAtBirth,
  )
where

import ConstantsAndVectors (criticalPoint, distanceCloseCritical, uniformDraws)
import Data.IntMap (size)
import Data.Maybe (fromJust)
import Data.Random (uniform)
import Data.Vector qualified as DV (Vector, cons, empty, fromList, last, map, singleton, snoc, (!), (++))
import DataTypes (CdfParameters (..), LogSize, RandomNumber, TimeSeries)
import DrawAlgorithm (drawCDF)
import UtilityFunctions (deltaFunc, fromTimeSeries, inverseCumSimulation, randomUniform, rootMaybe, toTimeSeries)

-- sizeAtBirth :: CdFParameters -> seriesLength -> counter -> timeseries
sizeAtBirth :: CdfParameters -> Int -> Int -> TimeSeries
sizeAtBirth param seriesLength n
  | n == seriesLength = toTimeSeries DV.empty
  | otherwise = toTimeSeries $ DV.cons (Just (xBirth param)) (fromTimeSeries $ sizeAtBirth param' seriesLength (n + 1))
  where
    --  otherwise = toTimeSeries (DV.singleton $ Just (xBirth param)) DV.++ sizeAtBirth param' seriesLength (n + 1)
    param' = param {xBirth = fromJust $ rootMaybe $ drawCDF param u}
    u = head $ randomUniform seed 1
    seed = 2 * (n + 1)

{-
define a default for counter (=length gammas + 1)
-}
-- simulationMatrix :: uniforms-> gammas -> seriesLength -> Counter -> matrix TimeSerieses

simulationMatrix :: DV.Vector RandomNumber -> DV.Vector Double -> Int -> Int -> DV.Vector TimeSeries
simulationMatrix _ _ _ (-1) = DV.empty
simulationMatrix us gammaValues seriesLength n = DV.singleton (sizeAtBirth param seriesLength 0) DV.++ simulationMatrix us gammaValues seriesLength (n - 1)
  where
    param =
      CdfParameters
        { -- xBirth = log $ inverseCumSimulation (deltaFunc (distanceCloseCritical DV.! n)) (uniformDraws DV.! n),
          xBirth = deltaFunc (distanceCloseCritical DV.! n),
          gammaValue = gammaValues DV.! n,
          hValue = 0.0
        }

{-
import System.Random
import qualified Data.Vector.Unboxed as V

processVector :: V.Vector Double -> Double
processVector vec = V.sum vec

generateRandomVector :: Int -> Double -> Double -> IO (V.Vector Double)
generateRandomVector n lower upper = do
  gen <- getStdGen
  let samples = replicate n $ randomR (lower, upper) gen
  return $ V.fromList samples

main :: IO ()
main = do
  randomVector <- generateRandomVector 10 0.0 1.0
  let result = processVector randomVector
  print result

-}