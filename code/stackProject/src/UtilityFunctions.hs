module UtilityFunctions where

-- cdfLogSpace x, xb, gamma -> result
cdfLogSpace :: Double -> Double -> Double -> Double
cdfLogSpace x xb gamma = 1 - 2 ** (-gamma) * exp (gamma * (-2 * exp (x) + exp (xb) - x + xb))
