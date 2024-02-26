# 70% train, 30% test
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

f = open('data/abalone.data', encoding='utf-8')
x_data = []
y_data = []
for i in f.readlines():
    data = i[:-1].split(',')
    x_data.append(data[:-1])
    y_data.append(data[-1])

x_data = np.array(x_data)
y_data = np.array(y_data)

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3)

print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

