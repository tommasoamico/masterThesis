#!/bin/bash

stack build
stack clean
stack exec simulationsHaskell-exe
python3 pythonFiltering.py

