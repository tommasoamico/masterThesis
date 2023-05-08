{-# LANGUAGE ImportQualifiedPost #-}

module DataTypes
  ( CdfParameters (..),
    TimeSeries (..),
    LogSize,
    BisectBracket,
    RandomNumber,
    Bracket,
    Size,
  )
where

import Data.Vector qualified as DV

-- implement the Default instance for CdfParameters (Also Functor?)
data CdfParameters = CdfParameters
  {xBirth :: Double, gammaValue :: Double, hValue :: Double}
  deriving (Show)

type LogSize = Double

type RandomNumber = Double

type BisectBracket = (Double, Double)

type Bracket = (Double, Double)

newtype TimeSeries = TimeSeries (DV.Vector (Maybe Double)) deriving (Show)

type Size = Double
