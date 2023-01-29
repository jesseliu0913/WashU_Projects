import numpy as np
from ReadData import read_data
import os

np.random.seed(913)


def var_exp_Uni(x, y, filename):
    data_length, feature_num = x.shape
    # filename = '../result/Text/Original/hyper_para_MSE.txt'
    r2_lst = []
    with open(filename) as f:
        file = f.read()
        weights = np.array([float(i) for i in (
            list(list(file.split('\n'))[0].replace('[', '').replace(']', '').replace(',', '').split(' '))[1:])])
        bias = np.array([float(i) for i in (
            list(list(file.split('\n'))[1].replace('[', '').replace(']', '').replace(',', '').split(' '))[1:])])

    for i in range(feature_num):
        y_pre = x[:data_length, i] * weights[i] + bias[i]

        mse = np.sum((y - y_pre) ** 2) / data_length
        # print(mse)
        var = np.sum((y_pre - y_pre.mean()) ** 2) / (data_length-280)
        # print(var)

        r2 = 1 - mse / var
        r2_lst.append(r2)

    # print(r2_lst)

    # with open('../result/Text/Original/Testdataset', 'a+') as f:
    #     f.write('univariate_MSE_r2: ')
    #     f.write(str(r2_lst))
    #     f.write('\n')

    return r2_lst

def var_exp_Unit(x, y, filename):
    data_length, feature_num = x.shape
    # filename = '../result/Text/Original/hyper_para_MSE.txt'
    r2_lst = []
    with open(filename) as f:
        file = f.read()
        weights = np.array([float(i) for i in (
            list(list(file.split('\n'))[0].replace('[', '').replace(']', '').replace(',', '').split(' '))[1:])])
        bias = np.array([float(i) for i in (
            list(list(file.split('\n'))[1].replace('[', '').replace(']', '').replace(',', '').split(' '))[1:])])

    for i in range(feature_num):
        y_pre = x[:data_length, i] * weights[i] + bias[i]

        mse = np.sum((y - y_pre) ** 2) / data_length
        # print(mse)
        var = np.sum((y_pre - y_pre.mean()) ** 2) / (data_length-20)
        # print(var)

        r2 = 1 - mse / var
        r2_lst.append(r2)

    # print(r2_lst)

    # with open('../result/Text/Original/Testdataset', 'a+') as f:
    #     f.write('univariate_MSE_r2: ')
    #     f.write(str(r2_lst))
    #     f.write('\n')

    return r2_lst

# data_x, data_y, traindata_x, traindata_y, testdata_x, testdata_y = read_data()
# r2_uni = var_exp_Uni(traindata_x, traindata_y)
# r2_uni = var_exp_Uni(testdata_x, testdata_y)


def var_exp_multi(x, y, filename):
    data_length, feature_num = x.shape
    # filename = '../result/Text/Original/hyper_para_MultiVariate_MSE.txt'
    r2_lst = []
    x0 = np.random.randint(1, 2, (data_length, 1))
    x = np.concatenate((x0, x), axis=1)

    with open(filename) as f:
        file = f.read()
        weight_bias = np.array([[float(i)] for i in (file.replace('[', '').replace(']', '').replace('\n', '').split(' ')[1:])])

    y_pre = np.matmul(x, weight_bias)
    y = y.reshape(data_length, 1)

    var = np.sum((y_pre - y_pre.mean()) ** 2) / (data_length-650)
    mse = np.sum((y - y_pre) ** 2) / data_length
    r2 = 1 - mse / var
    # print(r2)

    # with open('../result/Text/Norm/Testdataset', 'a+') as f:
    #     f.write('multivariate_MSE_r2: ')
    #     f.write(str(r2))
    #     f.write('\n')

    return r2

def var_exp_multit(x, y, filename):
    data_length, feature_num = x.shape
    # filename = '../result/Text/Original/hyper_para_MultiVariate_MSE.txt'
    r2_lst = []
    x0 = np.random.randint(1, 2, (data_length, 1))
    x = np.concatenate((x0, x), axis=1)

    with open(filename) as f:
        file = f.read()
        weight_bias = np.array([[float(i)] for i in (file.replace('[', '').replace(']', '').replace('\n', '').split(' ')[1:])])

    y_pre = np.matmul(x, weight_bias)
    y = y.reshape(data_length, 1)

    var = np.sum((y_pre - y_pre.mean()) ** 2) / data_length
    mse = np.sum((y - y_pre) ** 2) / data_length
    r2 = 1 - mse / var
    # print(r2)

    # with open('../result/Text/Norm/Testdataset', 'a+') as f:
    #     f.write('multivariate_MSE_r2: ')
    #     f.write(str(r2))
    #     f.write('\n')

    return r2

# x, y, traindata_x, traindata_y, testdata_x, testdata_y = read_data()
# r2_vari = var_exp_multi(testdata_x, testdata_y)
# r2_vari = var_exp_multi(traindata_x, traindata_y)
# 0.9407654263782664    0.8384389239241945v
# 0.8161596317900286
# 0.983039294321082

# -4.0599937412559886