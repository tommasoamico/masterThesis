{-# LANGUAGE ImportQualifiedPost #-}

module DataTypes
  ( CdfParameters (..),
    TimeSeries (..),
    LogSize,
    RandomNumber,
    Bracket,
    Size,
    SimulationParameters (..),
    SimulationParametersNotMonadic (..),
  )
where

import Data.Vector qualified as DV
import System.Random (StdGen)



data CdfParameters = CdfParameters
  {xBirth :: Double, gammaValue :: Double, hValue :: Double}
  deriving (Show)

data SimulationParameters = SimulationParameters
  {params:: CdfParameters, generator::StdGen, accumulatedDraw :: DV.Vector LogSize} deriving (Show)

data SimulationParametersNotMonadic = SimulationParametersNotMonadic
  {paramsNotMonadic:: CdfParameters, generatorNotMonadic::StdGen} deriving (Show)

type LogSize = Double

type RandomNumber = Double

type Bracket = (Double, Double)

type TimeSeries = DV.Vector LogSize

type Size = Double
