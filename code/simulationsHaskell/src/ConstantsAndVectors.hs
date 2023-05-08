module ConstantsAndVectors
  ( criticalPoint,
    distanceCloseCritical,
    gammaValuesCloseCritical,
    uniformDraws,
  )
where

import Data.Vector (Vector, fromList)
import DataTypes (RandomNumber)
import UtilityFunctions (distanceFromCritical, randomUniform)

criticalPoint :: Double
criticalPoint = 1 / log 2

distanceCloseCritical :: Vector Double
distanceCloseCritical = distanceFromCritical (lowerLimit, upperLimit) n
  where
    n = 100
    lowerLimit = 1e-2
    upperLimit = 1e-3

-- Active phase (we are below the critical concentration)
gammaValuesCloseCritical :: Vector Double
gammaValuesCloseCritical = (criticalPoint -) <$> distanceCloseCritical

uniformDraws :: Vector RandomNumber
uniformDraws = fromList $ randomUniform seed n
  where
    seed = 2
    n = length distanceCloseCritical
