import pandas as pd
import numpy as np
from ReadData import read_data
from PreProcess import standardizing
from PreProcess import normalizing
from Uni_Variant import uni_variant
from Multi_Variant import multi_variant
from DrawData import draw_data
from Test import var_exp_Uni
from Test import var_exp_multi
from Test import var_exp_Unit
from Test import var_exp_multit

np.random.seed(913)

if __name__ == '__main__':
    datafile = pd.read_excel("../data/Concrete_Data.xls", "Sheet1").to_numpy()

    # 0: raw data
    # 1: standardlize
    # 2: normalize
    print("show our data distribution plot")
    # draw_data(1)
    # draw_data(0)
    # draw_data(2)

    # 0: raw data
    # 1: standardlize
    # 2: normalize
    x, y, traindata_x, traindata_y, testdata_x, testdata_y = read_data(datafile, 1)

    '''
    Here we train our two kinds of Linear Regression Model and save the parameter
    Mention: I have generate the weight and bias after longt time training, i
    f you run the save file codes again, it will cover the formal one
    '''
    # Uni_variant
    # 1: MSE   else: MAE
    # weight, bias = uni_variant(traindata_x, traindata_y, 0)
    # with open("../result/Text/Original/hyper_para_MAE.txt", "w+") as f:
    #     f.write("univariate_MAE_weight: ")
    #     f.write(str(weight))
    #     f.write("\nunivariate_MAE_bias: ")
    #     f.write((str(bias)))
    #     f.close()

    # Multi_Variant
    # 1: MSE   else: MAE
    # weight_bias, final = multi_variant(traindata_x, traindata_y, 1)
    # with open("../result/Text/Standard/hyper_para_MultiVariate_MSE.txt", "w+") as f:
    #     f.write("Multi-Variate_raw_weight_bias: ")
    #     f.write(str(weight_bias))

    # weight_bias, final = multi_variant(traindata_x, traindata_y, 0)
    # with open("../result/Text/Norm/hyper_para_MultiVariate_MAE.txt", "w+") as f:
    #     f.write("Multi-Variate_raw_weight_bias: ")
    #     f.write(str(weight_bias))

    '''
        Here we test our two kinds of Linear Regression Model using r^2
    '''
    print("Start a test for our Model -------------------------------------------------------------------------")

    # print("The Original Dataset")
    # print("For the Uni-variant")
    # filename1 = '../result/Text/Standard/hyper_para_MSE.txt'
    # r2_uni_1 = var_exp_Uni(traindata_x, traindata_y, filename1)
    # r2_uni_2 = var_exp_Unit(testdata_x, testdata_y, filename1)
    # print("Train r2_uni is :", r2_uni_1)
    # print("Test r2_uni is :", r2_uni_2)
    #
    # filename2 = '../result/Text/Original/hyper_para_MultiVariate_MSE.txt'
    # print("For the Multi-variant")
    # r2_multi_1 = var_exp_multi(traindata_x, traindata_y, filename2)
    # r2_multi_2 = var_exp_multit(testdata_x, testdata_y, filename2)
    # print("Train r2_multi is :", r2_multi_1)
    # print("Test r2_multi is :", r2_multi_2)


    print("The Standard Dataset")
    print("For the Uni-variant")
    filename1 = '../result/Text/Original/hyper_para_MSE.txt'
    r2_uni_1 = var_exp_Uni(traindata_x, traindata_y, filename1)
    r2_uni_2 = var_exp_Unit(testdata_x, testdata_y, filename1)
    print("Train r2_uni is :", r2_uni_1)
    print("Test r2_uni is :", r2_uni_2)

    filename2 = '../result/Text/Standard/hyper_para_MultiVariate_MSE.txt'
    print("For the Multi-variant")
    r2_multi_1 = var_exp_multi(traindata_x, traindata_y, filename2)
    r2_multi_2 = var_exp_multit(testdata_x, testdata_y, filename2)
    print("Train r2_multi is :", r2_multi_1)
    print("Test r2_multi is :", r2_multi_2)

# Oct 12 retrain data
"""
Raw:
r2_uni is : [0.9983746743379894, 0.9971007080301754, 0.9096454465271572, 0.9771630063252803, 0.993347333679208, 0.9581522248361448, 0.9737175094881474, 0.998687897333502]
r2_multi is :  0.9987929565630864 / 0.9919332641239935
MAE: 0.9988012593802981 / 0.9917845449088131

Train r2_multi is : 0.9976698756889176
Test r2_multi is : 0.9840222933918386
"""

"""
Stand:
r2_uni is : [0.9988099076240481, 0.9978883470697861, 0.9972278555908506, 0.9966255261333462, 0.9984990884202306, 0.9970760776687233, 0.9970733755073725, 0.9983567741686615]
Train r2_multi is : 0.9985997817454286
Test r2_multi is : 0.9903727161523553
"""

"""
Norm
r2_uni is : [0.9896430664100385, 0.9870628199709967, 0.9852422735027107, 0.9835949297444635, 0.9888039630857871, 0.9848842861049224, 0.9846791410065089, 0.9879462743203834]
Train r2_multi is : 0.9985997817454286
Test r2_multi is : 0.9903727161523553

"""

# UINI
# wrong type
"""
Norm
Train r2_multi is : [0.9896430664100385, 0.9870628199709967, 0.9852422735027107, 0.9835949297444635, 0.9888039630857871, 0.9848842861049224, 0.9846791410065089, 0.9879462743203834]
Test r2_multi is : [0.9327263584994452, 0.9092933917341293, 0.8985371855128446, 0.9013041199796805, 0.9192532268122964, 0.8919873953048549, 0.8998381139034274, 0.9397974063895898]
"""

"""
Original
Train r2_multi is : [0.9988701257699945, 0.9980706712789523, 0.9975377358146177, 0.9970518269382391, 0.9985955968068272, 0.9974164023425955, 0.997415729083013, 0.9984694093022537]
Test r2_multi is : [0.9928602693495726, 0.9869660150915381, 0.9831001467739692, 0.9836816450392147, 0.989925284061411, 0.981766710358838, 0.9814580639958934, 0.9910851981305051]
"""

"""
Standard
Train r2_multi is : [0.9988099076240481, 0.9978883470697861, 0.9972278555908506, 0.9966255261333462, 0.9984990884202306, 0.9970760776687233, 0.9970733755073725, 0.9983567741686615]
Test r2_multi is : [0.9925147757129426, 0.9857724682113408, 0.9810604993048115, 0.9814267047729804, 0.9892499082757545, 0.9794022121944383, 0.9792021063488862, 0.9906034005340841]
"""








