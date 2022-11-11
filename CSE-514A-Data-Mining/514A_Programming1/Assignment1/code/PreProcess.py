import numpy as np



def standardizing(data):
    std_data = np.std(data, axis=0)
    mu_data = np.mean(data, axis=0)
    result = (data - mu_data) / std_data
    return result


def normalizing(data):
    max = np.max(data, axis=0)
    min = np.min(data, axis=0)
    range = max - min
    result = (data - min) / range
    return result