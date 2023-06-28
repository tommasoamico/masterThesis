module UtilityFunctions
  ( cdfLogSpace,
    getBracket,
    startingPointLog,
  )
where

import Data.Vector (Vector, fromList)
import DataTypes (Bracket, CdfParameters (..), LogSize, RandomNumber, Size, TimeSeries (..))
import Numeric.RootFinding (Root (..))
import System.Random (StdGen, randomR)

cdfLogSpace :: CdfParameters -> LogSize -> Double
cdfLogSpace param x = 1 - exp((-2* exp x  + exp xb )*gamma) * ((2* exp x  + h)**((-1 + h) * gamma))*(( exp xb  + h)**(gamma - h*gamma))
  where
    xb = xBirth param
    gamma = gammaValue param
    h = hValue param

{-
-- createLinspace lowerBound upperBound numberOfPoints -> result
createLinspace :: Double -> Double -> Int -> [Double]
createLinspace _ _ n | n <= 0 = error "Number of points must be positive"
createLinspace lowerBound upperBound n = [lowerBound, lowerBound + step .. upperBound]
  where
    step = (upperBound - lowerBound) / fromIntegral (n - 1)

createLogspace :: Double -> Double -> Int -> [Double]
createLogspace _ _ n | n <= 0 = error "Number of points must be positive"
createLogspace lowerBound _ _ | lowerBound <= 0 = error "lowerBound must be positive"
createLogspace lowerBound upperBound n = map (10 **) linspace
  where
    linspace = createLinspace (log10 lowerBound) (log10 upperBound) n
    log10 = logBase 10
-}



-- inverseCum :: RandomNumber -> Size
inverseCum :: RandomNumber -> Size
inverseCum unif = 10 * unif ** (1.0 / 0.1)



-- getBracket :: xBirth -> Bracket
getBracket :: Double -> Bracket
getBracket xb | xb <= -7 = (leftBracket, 10)
              | otherwise = (-25, 10)
  where
    -- - log (2.0 * (1.0 - 0.5)) + xb + exp xb - 2.0
    leftBracket = xb + exp xb - 2.0
    


-- startingPointLog :: h -> gen -> Double
startingPointLog :: Double -> StdGen -> LogSize
startingPointLog h gen | signum h == 0 = log $ inverseCum $ fst $ randomR (0, 1) gen
                    | otherwise = log h