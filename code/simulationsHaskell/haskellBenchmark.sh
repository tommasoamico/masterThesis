#!/bin/bash
nameFile=$"/Users/tommaso/Desktop/masterThesis/data/codeBenchMark/haskell/runTime1m.txt"
for i in {1..50}
do
time=$(stack exec simulationsHaskell-exe)
stack clean
echo "$time" >> "$nameFile"
echo "$i iterations completed"
done