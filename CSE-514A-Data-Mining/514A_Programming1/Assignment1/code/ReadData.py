import numpy as np
import pandas as pd
from PreProcess import standardizing
from PreProcess import normalizing
from sklearn import preprocessing



def read_data(datafile, type):
    # read the data file and shuffle them
    # datafile = pd.read_excel("../data/Concrete_Data.xls", "Sheet1").to_numpy()
    np.random.shuffle(datafile)
    if type == 1:
        datafile = standardizing(datafile)
    if type == 2:
        datafile = normalizing(datafile)
    else:
        datafile = datafile

    # test stand_data effect
    # preprocessing data use standardizing or normalizing
    # std = preprocessing.StandardScaler()
    # daatfile = std.fit_transform(datafile)

    #  split dataset to train / test
    traindata = datafile[:900, :]
    traindata_y = traindata[:, -1]
    traindata_x = traindata[:, :-1]

    testdata = datafile[900:, :]
    testdata_y = testdata[:, -1]
    testdata_x = testdata[:, :-1]

    data_y = datafile[:, -1]
    data_x = datafile[:, :-1]

    return data_x, data_y, traindata_x, traindata_y, testdata_x, testdata_y

# read_data()
