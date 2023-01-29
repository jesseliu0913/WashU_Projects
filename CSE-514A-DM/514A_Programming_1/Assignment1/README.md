# 514A Programming1-Linear Regression

Here is my first Programming Assignment in 514A Data Mining in WashU 2022Fall, it mainly contain how to use Uni-Variant and Multi-Variant to complete Linear Regression, our dataset from UCI [Concrete Compressive Strength dataset](https://archive.ics.uci.edu/ml/datasets/Concrete+Compressive+Strength)

## Requirements

numpy == 1.21.6
pandas == 13.35
matplotlib == 3.5.3

## File Directory

514A_Programming
--Assignment1
------code
----------Config.py
----------DrawData.py
----------main.py
----------Multi_Variant.py
----------Preprocess.py
----------ReadData.py
----------Test.py
----------Uni_Variant.py
----data
----------Concrete_Data.xls
----result
----------Image
----------Test


## Run
In the main.py file, there conclude all command and function we need, just open the mian.py file and run it. Importantly, the Uni-Variant and Multi-Variant Output is refresh every time you run it, and the parameters depends on what you have set in the Config.py file, but the Test code, the calculation about r^2 depends on the data from the previous parameter I have trained for a long time and save them in the /Result/Text directory.
Actually, you can use the save file code which I have commented to save the weight and bias parameters generated with your own hyper-parameters.

Config.py: conclude all hyper-parameters in Uni-Variant and Multi_Variant

ReadData.py: read the /data/Concrete_Data.xls file and split them into Train set and Test set.

PreProcess.py: Here are two function about Standardizing and Normalizing dataset.

Uni-Variant.py: Achieve the Uni-Variant Linear Regression with MSE and MAE loss function, you can change it in the main.py file

Multi-Variant.py: Achieve the Multi_Variant Linear Regression with MSE and MAE loss function, you can change it in the main.py file

Test.py: Test both Uni-Variant and Multi-Variant model via calculated Variance Explained ( $R^2$)


