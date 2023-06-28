{-# LANGUAGE ImportQualifiedPost #-}


module Main where

import ConstantsAndVectors (gammaSim, hSim, lengthSimulation, simulationParams, simulationParamsNotMonadic, hSimList)
import Data.ByteString.Lazy qualified as BL
import Data.Csv (FromRecord, ToField, ToRecord, encode, toField, toRecord)
import Data.IntMap (fromList)
import Data.Vector qualified as DV (Vector, fromList, map, toList, (!), empty, snoc)
import DataTypes qualified as DT
import SimulationFunctions (simulate, simulateNotMonadic)
import DrawAlgorithm (drawCDF)
import System.Random (mkStdGen)
import Numeric.RootFinding (Root, fromRoot, Root(..))
import Data.Time (diffUTCTime, getCurrentTime)
import UtilityFunctions (startingPointLog)
import DataTypes (LogSize)





-- instance ToField a => ToRecord (DT.TimeSeries) where
-- toRecord (DT.TimeSeries vector) = DV.map toField vector

{-
vector :: [Maybe Double]
vector = DV.toList $ fromTimeSeries $ sizeAtBirth param 500 0
  where
    param =
      DT.CdfParameters
        { DT.xBirth = 1.0,
          DT.gammaValue = 1.0,
          DT.hValue = 0.0
        }
-}

{-
matrix :: DV.Vector DT.TimeSeries
matrix = simulationMatrix uniformDraws gammaValuesCloseCritical 500 99

transformedMatrix :: [[Maybe Double]]
transformedMatrix = DV.toList $ DV.map (DV.toList . fromTimeSeries) matrix

csvData :: BL.ByteString
csvData = encode transformedMatrix

main :: IO ()
main = do
  BL.writeFile "test.csv" csvData
-}
--saveTimeseries::[DV.Vector Double]



-- single value
{-
main :: IO ()
main = do 
  start <- getCurrentTime
  -- Root DV.Vector Double
  let timeSeries = simulate lengthSimulation (simulationParams $ mkStdGen 1)
  --let outOfMonad = fromRoot DV.empty timeSeries
  -- Root [DV.Vector Double]
  let listSave = (\x -> [x]) <$> timeSeries
  -- Root ByteString
  let csvData = encode <$> listSave
  --Bytesryng -> IO () | Root ByteString
  -- Root IO ()
  let byteData = BL.writeFile "test.csv" <$> csvData
  case byteData of
    Root x -> x
    NotBracketed -> putStrLn "NotBracketed"
    SearchFailed -> putStrLn "SearchFailed"
  end <- getCurrentTime
  print $ diffUTCTime end start
-}

-- listOfValues
mainListFunction :: [Double] -> DV.Vector (DV.Vector LogSize)  -> IO ()
mainListFunction [] vector = do
  -- [DV.Vector DV.Vector Double]
  -- Root ByteString
  let csvData = encode $ DV.toList vector
  -- Root IO ()
  BL.writeFile "test.csv"  csvData
mainListFunction (h:hs) vector = do
  let gen = mkStdGen $ length hs
  let params = (DT.SimulationParameters {DT.params = DT.CdfParameters {DT.xBirth = startingPointLog h gen, DT.gammaValue = gammaSim, DT.hValue = h}, DT.generator = gen, DT.accumulatedDraw = DV.empty})
  -- Root DV.Vector Double
  let timeSeries = simulate lengthSimulation params  
  -- Root DV.Vector DV.Vector Double
  let newVector = DV.snoc vector <$> timeSeries
  case newVector of
    Root x -> mainListFunction hs x
    NotBracketed -> putStrLn $ mconcat ["NotBracketed, ", show $ length hs, " h values remaining"] 
    SearchFailed -> putStrLn $ mconcat["SearchFailed", show $ length hs, " h values remaining"]

main::IO()
main = mainListFunction hSimList DV.empty

  
--notMonadic
{-
main :: IO ()
main = do
  start <- getCurrentTime
  let timeSeries = simulateNotMonadic lengthSimulation (simulationParamsNotMonadic $ mkStdGen 1) 
  let listSave = [timeSeries]
  let csvData = encode listSave
  BL.writeFile "test.csv" csvData
  end <- getCurrentTime
  print $ diffUTCTime end start
-}