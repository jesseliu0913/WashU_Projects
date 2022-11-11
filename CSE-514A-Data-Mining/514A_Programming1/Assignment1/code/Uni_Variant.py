import numpy as np
from ReadData import read_data
import matplotlib.pyplot as plt
import Config


def MAE(y_pre, y, x, weight, bias, learning_rate):
    data_length  = len(y)
    loss = np.sum(np.abs(y - y_pre)) / data_length
    partial = np.where(y >= y_pre, -1, 1)

    partial_weight = np.sum(partial * x) / data_length
    weight -= learning_rate * partial_weight

    partial_bias = 1.0 / data_length * np.sum(partial)
    bias -= learning_rate * partial_bias

    return loss, weight, bias

def MSE(y_pre, y, x, weight, bias, learning_rate):
    # mis = np.where(y_pre != traindata_y)[0][1]
    # mis = np.where(y_pre != traindata_y)[0]
    data_length = len(y)
    loss = np.sum((y - y_pre) ** 2) / data_length
    gap = y - y_pre

    partial_weight = np.sum((-2 * gap * x)) / data_length
    weight -= learning_rate * partial_weight

    partial_bias = np.sum((-2 * gap)) / data_length
    bias -= learning_rate * partial_bias

    return loss, weight, bias


def uni_variant(traindata_x, traindata_y, type):
    data_length, feature_num = traindata_x.shape
    weight_lst = []
    bias_lst = []
    iter = Config.epoch_uni
    # 20000
    y = traindata_y

    for i in range(feature_num):
        weight = Config.weight_uni
        bias = Config.bias_uni

        learning_rate = Config.lr_uni
        count = 1
        previous_loss = 0
        x = traindata_x[:, i]


        for j in range(iter):
            y_pre = x * weight + bias

            if type == 1:
                loss, weight, bias = MSE(y_pre, traindata_y, x, weight, bias, learning_rate)
            else:
                loss, weight, bias = MAE(y_pre, traindata_y, x, weight, bias, learning_rate)

            count += 1
            if previous_loss - loss < 1e-4:
                learning_rate = learning_rate * 1e-1
            if previous_loss == loss:
                break

            # print(loss)
            previous_loss = loss

        weight_lst.append(weight)
        bias_lst.append(bias)
        final = weight * x + bias


        print("all of the feature will be show")
        plt.scatter(traindata_x[:, i], traindata_y)
        plt.title("univariate_MSE_data_Standardizing")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.plot(traindata_x[:, i], final, color="yellow")


        # plt.savefig(f"../result/Image/Ori_univariate_MSE_Stand/univariate_MSE{i}.png")
        # plt.clf()
        plt.show()
        print("-------------------------------------------------------------------------------------")
        print(f"Here is feature {i}")
        print("Finish Uni-Variant Linear Regression")
        print("Here is weight : ", weight)
        print("Here is a bias : ", bias)

    return weight_lst, bias_lst

# data_x, data_y, traindata_x, traindata_y, testdata_x, testdata_y = read_data()
# weight, bias = uni_variant(traindata_x, traindata_y)
# with open("../result/Text/Original/hyper_para_MSE.txt", "w+") as f:
#     f.write("univariate_MSE_weight: ")
#     f.write(str(weight))
#     f.write("\nunivariate_MSE_bias: ")
#     f.write((str(bias)))
#     f.close()
# print(weight)
# print(bias)
