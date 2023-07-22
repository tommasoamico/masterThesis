{-# LANGUAGE ImportQualifiedPost #-}


module Main where

import ConstantsAndVectors (gammaSim, hSim, lengthSimulation, simulationParams, simulationParamsNotMonadic, hSimList, savePath)
import Data.ByteString.Lazy qualified as BL
import Data.Csv (FromRecord, ToField, ToRecord, encode, toField, toRecord, record)
import Data.Sequence ((|>), empty, Seq)
import Data.Vector qualified as DV
import DataTypes qualified as DT
import SimulationFunctions (simulate, simulateNotMonadic)
import DrawAlgorithm (drawCDF)
import System.Random (mkStdGen)
import Numeric.RootFinding (Root, fromRoot, Root(..))
import Data.Time (diffUTCTime, getCurrentTime)
import UtilityFunctions (startingPointLog, encodeSequence)
import Data.Foldable (toList)
import DataTypes (LogSize)
import Control.DeepSeq (deepseq)




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


{-
-- single value
main :: IO ()
main = do

  let params = simulationParams $ mkStdGen 0
  -- Root DV.Vector Double
  start <- getCurrentTime
  let timeSeries = simulate lengthSimulation params
  end <- timeSeries `deepseq` getCurrentTime
  print $ diffUTCTime end start
{-
  let byteData = encodeSequence savePath <$> timeSeries

  case byteData of
    Root x -> x
    NotBracketed -> putStrLn "NotBracketed"
    SearchFailed -> putStrLn "SearchFailed"
-}
-}



-- listOfValues
-- mainListFunction hValues, 
mainListFunction :: [Double]  -> IO ()
mainListFunction [] = print "Done"
mainListFunction (h:hs) = do
  start <- getCurrentTime
  let gen = mkStdGen $ length hs
  let params = (DT.SimulationParameters {DT.params = DT.CdfParameters {DT.xBirth = startingPointLog h gen, DT.gammaValue = gammaSim, DT.hValue = h}, DT.generator = gen, DT.accumulatedDraw = empty})
  -- Root DV.Vector Double
  let timeSeries = simulate lengthSimulation params
  -- IO ()
  let byteData = encodeSequence savePath <$> timeSeries
  case byteData of
    Root x ->  x
    NotBracketed -> putStrLn $ mconcat ["NotBracketed, ", show $ length hs, " h values remaining"]
    SearchFailed -> putStrLn $ mconcat ["SearchFailed", show $ length hs, " h values remaining"]
  end <- getCurrentTime
  putStrLn $ mconcat ["Execution Done in ", show $ diffUTCTime end start, show $ length hs, "  h values remaining "]
  mainListFunction hs

main::IO()
main = mainListFunction hSimList

