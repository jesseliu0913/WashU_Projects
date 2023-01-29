import random
import numpy as np

from ReadData import read_data
import matplotlib.pyplot as plt
import Config

np.random.seed(913)




def multi_variant(traindata_x, traindata_y, type):

    x0 = np.random.randint(1, 2, (900,1))
    x = np.concatenate((x0, traindata_x), axis=1)

    data_length, feature_num = traindata_x.shape
    iter = Config.epoch_multi

    weight = np.random.rand(feature_num + 1, 1) - 0.5
    # bias = np.random.rand(data_length, 1)
    learning_rate = Config.lr_multi
    previous_loss = 0

    traindata_y = traindata_y.reshape(900, 1)
    for j in range(iter):
        y_pre = np.matmul(x, weight)
        if type == 1:
            y_pre = np.matmul(x, weight)
            gap = traindata_y - y_pre
            loss = np.sum((traindata_y - y_pre) ** 2) / data_length
            grad = -1 * np.matmul(gap.reshape(1, -1), x)
            weight -= learning_rate * grad.reshape(-1, 1)
            print(loss)
        else:

            loss = np.sum(np.abs(traindata_y - y_pre)) / data_length

            grad = np.matmul(1.0 / data_length * np.where(traindata_y >= y_pre, -1, 1).reshape(1, -1), x)
            weight -= learning_rate * grad.reshape(9, 1)
            print(loss)

        if previous_loss - loss < 1e-8:
            learning_rate = learning_rate * 1e-1

    final = np.matmul(x, weight)


    return weight, final


# traindata_x, traindata_y, testdata_x, testdata_y = read_data()
# weight_bias, final = multi_variant(traindata_x, traindata_y, testdata_x, testdata_y)
#
#
#
# with open("../result/hyper_para_Multi-Variate.txt", "w+") as f:
#     f.write("Multi-Variate_raw_weight_bias: ")
#     f.write(str(weight_bias))



