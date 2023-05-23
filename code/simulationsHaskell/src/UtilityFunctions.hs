module UtilityFunctions
  ( cdfLogSpace,
    createLinspace,
    createLogspace,
    distanceFromCritical,
    linearCongruentualGenerator,
    randomUniform,
    rootMaybe,
    deltaFunc,
    inverseCumSimulation,
    toTimeSeries,
    fromTimeSeries,
  )
where

import Data.Vector (Vector, fromList)
import DataTypes (Bracket, CdfParameters (..), LogSize, RandomNumber, Size, TimeSeries (..))
import Numeric.RootFinding (Root (..))

cdfLogSpace :: CdfParameters -> LogSize -> Double
cdfLogSpace param x = 1 - exp (-2 * exp x + exp xb) * gamma * ((2 * exp x + h) ** ((-1 + h) * gamma)) * ((exp xb + h) ** (gamma - h * gamma))
  where
    xb = xBirth param
    gamma = gammaValue param
    h = hValue param

{-
cdfLogSpace :: CdfParameters -> LogSize -> Double
cdfLogSpace param x = 1 - 2 ** (-gamma) * gamma * exp (gamma * (-x + xb + exp xb - 2 * (exp x)) + log (1 + 2 * exp x))
  where
    xb = xBirth param
    gamma = gammaValue param
-}

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

distanceFromCritical :: Bracket -> Int -> Vector Double
distanceFromCritical _ n | n <= 0 = error "Number of points must be positive"
distanceFromCritical bracket n = fromList $ uncurry createLogspace bracket n

-- uniformRandomGenerator m a c x0 nToTake = (a * x0 + c) `mod` m
linearCongruentualGenerator :: Int -> Int -> Int -> Int -> Int -> [Int]
linearCongruentualGenerator _ _ _ _ nToTake
  | nToTake < 0 = error "Number of points must be positive"
linearCongruentualGenerator m a c x0 nToTake
  | nToTake > 0 = partialList ++ [a * last partialList `mod` m]
  | otherwise = [x0]
  where
    partialList = linearCongruentualGenerator m a c x0 (nToTake - 1)

{-
Provide a default fo this function
-}
-- uniformDraw seed (length list)
randomUniform :: Int -> Int -> Vector RandomNumber
randomUniform seed n = fromList $ tail $ map ((/ fromIntegral (m - 1)) . fromIntegral) (linearCongruentualGenerator m a c seed (n + 1))
  where
    m = 2 ^ 31 - 1
    a = 7 ^ 5
    c = 0

rootMaybe :: Root a -> Maybe a
rootMaybe (Root x) = Just x
rootMaybe _ = Nothing

-- delta :: DistanceFromCritical -> delta
deltaFunc :: Double -> Double
deltaFunc distance = (0.76 :: Double) * distance

-- inverseCum :: upperBound -> delta -> RandomNumber -> Size
inverseCum :: Double -> Double -> RandomNumber -> Size
inverseCum upperBound delta u = upperBound * u ** (1 / delta)

-- inverseCumSimulation delta -> RandomNumber -> logSize
inverseCumSimulation :: Double -> RandomNumber -> LogSize
inverseCumSimulation delta u = log10 $ inverseCum 10 delta u
  where
    log10 = logBase 10

toTimeSeries :: Vector (Maybe Double) -> TimeSeries
toTimeSeries = TimeSeries

fromTimeSeries :: TimeSeries -> Vector (Maybe Double)
fromTimeSeries (TimeSeries x) = x