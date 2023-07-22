import numpy as np
import pandas as pd

arrayFull:np.ndarray = np.array(pd.read_csv('/Users/tommaso/Desktop/masterThesis/data/simulationHaskell/simulation2/simulation2.csv'))

idx1 =[50000 + i*200 for i in range(350000//200)]
idx2 =[50000 + (i * 200) + 70 for i in range(350000//200)]

finalArray :np.ndarray= np.concatenate([arrayFull[:,j:z] for j, z in zip(idx1, idx2)], axis = 1)

np.savetxt('/Users/tommaso/Desktop/masterThesis/data/simulationHaskell/simulation2/simulation2.csv', finalArray, delimiter=',')