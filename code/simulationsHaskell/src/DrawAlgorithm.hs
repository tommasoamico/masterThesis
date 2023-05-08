module DrawAlgorithm (drawCDF) where

import Data.Default.Class (def)
import DataTypes (CdfParameters (..), LogSize, RandomNumber)
import Numeric.RootFinding (Root, ridders)
import UtilityFunctions (cdfLogSpace)

-- CdfParameters {} {xBirth :: Double, gammaValue :: Double, hValue :: Double}

-- functionToInvert CdfParameters unif x
functionToInvert :: CdfParameters -> Double -> LogSize -> Double
functionToInvert param unif x = cdfLogSpace param x - unif

{-
Insert right types + assure correct lower and upper bounds
-}
drawCDF :: CdfParameters -> RandomNumber -> Root LogSize
drawCDF param unif = ridders def (-400, 40) (functionToInvert param unif)
