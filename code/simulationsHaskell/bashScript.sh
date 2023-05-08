start_ns=$(date +%s%N)
stack exec simulationHaskell-exe
end_ns=$(date +%s%N)
elapsed_ms=$(((end_ns - start_ns) / 1000000))
