module DrawAlgorithm (drawCDF) where

--import ConstantsAndVectors (myRiddersParam)
import Data.Default.Class (def)
import DataTypes (CdfParameters (..), LogSize, RandomNumber)
import Numeric.RootFinding (Root, ridders)
import UtilityFunctions (cdfLogSpace, getBracket)
import ConstantsAndVectors (bracketPositiveH)

-- CdfParameters {xBirth :: Double, gammaValue :: Double, hValue :: Double}

-- functionToInvert CdfParameters unif x
functionToInvert :: CdfParameters -> RandomNumber -> LogSize -> Double
functionToInvert param unif x = cdfLogSpace param x - unif


{-
This version of draw CDF has overlapping patterns, to review if we want to reason about the program 
-}
drawCDF :: CdfParameters -> RandomNumber -> Root LogSize
drawCDF param unif | hValue param > 0.0 = ridders def bracketPositiveH $ functionToInvert param unif
                    | xBirth param > -20 = ridders def bracket $ functionToInvert param unif
                    | otherwise = pure $ - log 2 - (1 / gammaValue param) * log unif + xBirth param 
  where
    bracket = getBracket $ xBirth param


                    