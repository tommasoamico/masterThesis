#!/bin/bash
nameFile=$"/Users/tommaso/Desktop/masterThesis/data/codeBenchMark/python/runTime1m.txt"
for i in {1..50}
do
time=$(python3 runSimulations.py)
echo "$time" >> "$nameFile"
echo "$i iterations completed"
done