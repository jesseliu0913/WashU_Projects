import numpy as np

animal = [0.343, 0.451, 0.02, 0.078, 0.049, 0.029, 0.029]
small = [0.250, 0.255, 0.286, 0.077, 0.100, 0.250, 0.125]
yellow = [0.122, 0.269, 0.250, 0.214, 0.091, 0.111, 0.111]
warm = [0.921, 0.898, 0.600, 0.182, 0.250, 0.667, 0.333]
month = [0.154, 0.340, 0.167, 0.167, 0.222, 0.286, 0.286]
dog = [0.255, 0.269, 0.898, 0.340]
P_dog = 0.451


sum1 = 0
for i, j in zip(animal, small):
    result = i * j
    sum1 += result
print(sum1)

sum2 = 0
for i, j in zip(animal, yellow):
    result = i * j
    sum2 += result
print(sum2)

sum3 = 0
for i, j in zip(animal, warm):
    result = i * j
    sum3 += result
print(sum3)

sum4 = 0
for i, j in zip(animal, month):
    result = i * j
    sum4 += result
print(sum4)


mul = 1
for i in dog:
    mul *= i

numerator = mul * P_dog
print(numerator)
dominator = sum1 * sum2 * sum3 * sum4
# dominator = 0.19575399999999998 * 0.22825600000000001 * 0.788347 * 0.24999400000000002
print("dominator", dominator)
print(numerator / dominator)


animal = [58.4, 76.8, 6.2, 6, 76.3, 22.9, 4.7]
ani_num = np.sum(animal)
animal_prop = []
for i in animal:
    cat = i / ani_num
    animal_prop.append(cat)
print(animal_prop)
