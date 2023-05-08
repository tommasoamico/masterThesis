module ConstantsAndVectors (criticalPoint) where

criticalPoint :: Double
criticalPoint = 1 / log 2

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