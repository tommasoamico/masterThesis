{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE ImportQualifiedPost #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE UndecidableInstances #-}

module Main where

import ConstantsAndVectors (gammaValuesCloseCritical, uniformDraws)
import Data.ByteString.Lazy qualified as BL
import Data.Csv (FromRecord, ToField, ToRecord, encode, toField, toRecord)
import Data.IntMap (fromList)
import Data.Vector qualified as DV (Vector, fromList, map, toList, (!))
import DataTypes qualified as DT
import SimulationFunctions (simulationMatrix, sizeAtBirth)
import UtilityFunctions (fromTimeSeries, toTimeSeries)

-- instance ToRecord DT.TimeSeries

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
matrix :: DV.Vector DT.TimeSeries
matrix = simulationMatrix uniformDraws gammaValuesCloseCritical 500 99

transformedMatrix :: [[Maybe Double]]
transformedMatrix = DV.toList $ DV.map (DV.toList . fromTimeSeries) matrix

csvData :: BL.ByteString
csvData = encode transformedMatrix

main :: IO ()
main = do
  BL.writeFile "test.csv" csvData
