{-# LANGUAGE ImportQualifiedPost #-}

module DataTypes
  ( CdfParameters (..),
    TimeSeries,
    LogSize,
    RandomNumber,
    Bracket,
    Size,
    SimulationParameters (..),
    SimulationParametersNotMonadic (..),
  )
where

import Data.Sequence qualified as S
import System.Random (StdGen)




data CdfParameters = CdfParameters
  {xBirth :: Double, gammaValue :: Double, hValue :: Double}
  deriving (Show)

data SimulationParameters = SimulationParameters
  {params:: CdfParameters, generator::StdGen, accumulatedDraw :: S.Seq LogSize} deriving (Show)

data SimulationParametersNotMonadic = SimulationParametersNotMonadic
  {paramsNotMonadic:: CdfParameters, generatorNotMonadic::StdGen} deriving (Show)

type LogSize = Double

type RandomNumber = Double

type Bracket = (Double, Double)

type TimeSeries = S.Seq LogSize

type Size = Double
