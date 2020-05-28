import math
import numpy as np
import pandas as pd
from tqdm import tqdm

def get_test_data(data_test, len_test, seq_len, normalise, debug=False, only_close=False):
    '''
    Create x, y test data windows
    When passing only_close=False we generate another output sized as input 
    Warning: batch method, not generative, make sure you have enough memory to
    load data, otherwise reduce size of the training split.
    '''
    data_windows = []
    for i in range(len_test - seq_len):
        data_windows.append(data_test[i:i+seq_len])

    data_windows = np.array(data_windows).astype(float)
    real_data = data_windows
    data_windows = normalise_windows(data_windows, single_window=False) if normalise else data_windows

    x = data_windows[:, :-1]
    y = data_windows[:, -1, [0]]
    if not only_close: y = data_windows[:, -1]

    return x, y, real_data[:, -1, [0]]

def normalise_windows(window_data, single_window=False):
    '''Normalise window with a base value of zero'''
    normalised_data = []
    window_data = [window_data] if single_window else window_data

    for i in tqdm(range(len(window_data))):
        window = window_data[i]
        normalised_window = []
        for col_i in range(window.shape[1]):
            normalised_col = [((float(p) / (float(window[0, col_i]) + 0.00000001) ) - 1) for p in window[:, col_i]]
            normalised_window.append(normalised_col)
        normalised_window = np.array(normalised_window).T # reshape and transpose array back into original multidimensional format
        normalised_data.append(normalised_window)
    return np.array(normalised_data)


def get_y_test_data(data_test, len_test, seq_len, normalise, close_col=0, debug=False):
    '''
    Get y test data in memory
    (Doing this because x is too large and I just need the y values)
    '''
    y = []
    for i in range(len_test - seq_len):
        if debug: print("Mounting y ", i + 1, " of ", len_test - seq_len)
        y.append(data_test[i+seq_len][0])

    y = np.array(y)
    
    first_value = y[0]
    y = np.true_divide(y, first_value) if normalise else y # Normalize

    if debug: print("Finished y")
    return y