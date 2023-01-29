import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
from PreProcess import standardizing
from PreProcess import normalizing
from sklearn import preprocessing


def draw_data(type):
    datafile = pd.read_excel("../data/Concrete_Data.xls", "Sheet1").to_numpy()
    # minmax = preprocessing.minmax_scale(datafile)
    # std = preprocessing.StandardScaler()
    # daatfile = std.fit_transform(datafile)
    if type == 1:
        datafile = standardizing(datafile)
    if type == 2:
        datafile = normalizing(datafile)
    else:
        datafile = datafile

    plt.hist(datafile, bins=5, facecolor="blue", edgecolor="black")

    plt.xlabel("Interval")
    plt.ylabel("Frequency")
    plt.title("Distribution Histograms - Standardizing Data ")
    plt.show()