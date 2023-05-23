module DrawAlgorithm (drawCDF) where

import ConstantsAndVectors (myRiddersParam)
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
drawCDF param unif
  | signum (functionToInvert param unif lower) * signum (functionToInvert param unif upper) /= -1 = error "Root not bracketed"
  | otherwise = ridders myRiddersParam (lower, upper) (functionToInvert param unif)
  where
    -- lower = -log 2 - 2 + xBirth param + exp (xBirth param)
    -- upper = lower + 50
    lower = -900 :: Double
    upper = 20 :: Double
