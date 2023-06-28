import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np


def real_data_loading(data: np.ndarray, seq_len):
    """Load and preprocess real-world datasets.
    Args:
      - data_name: Numpy array with the values from a a Dataset
      - seq_len: sequence length

    Returns:
      - data: preprocessed data.
    """

    # Normalize the data
    scaler = MinMaxScaler().fit(data)
    ori_data = scaler.transform(data)

    # Preprocess the dataset
    temp_data = []
    # Cut data by sequence length
    for i in range(0, len(ori_data) - seq_len):
        _x = ori_data[i:i + seq_len]
        temp_data.append(_x)

    # Mix the datasets (to make it similar to i.i.d)
    idx = np.random.permutation(len(temp_data))
    data = []
    for i in range(len(temp_data)):
        data.append(temp_data[idx[i]])
    return data


def transformationsDf(path, seq_len: int):
    timeSeriesDf: pd.DataFrame = pd.read_csv(path)
    # Data transformations to be applied prior to be used with the synthesizer model
    processed_data = real_data_loading(
        timeSeriesDf.values, seq_len=seq_len)

    return processed_data


def transformationsArray(array: np.ndarray, seq_len: int):
    if len(array.shape) == 1:
        processArray = array.reshape(-1, 1)
    else:
        processArray = array.T

    processed_data = real_data_loading(
        data=processArray, seq_len=seq_len)

    return processed_data
