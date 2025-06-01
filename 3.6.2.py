#Метод главных компонент PCA
#(principal component analysis)- алгоритм без учителя
#PCA - часто используют для понижения размерности 


#Задача состоит в выяснении завасимости между признаками
#PCA выполняется качественная оценка этой зависимости путем поиска главных осей координат и их использования для описания набора данных


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
iris = sns.load_dataset('iris')

# sns.pairplot(iris,hue='species')

data = iris[['petal_width', 'petal_length','species']]

data_v = data[data['species']=='versicolor']
data_v=data_v.drop(columns='species')
X= data_v['petal_width']
Y = data_v['petal_length']

plt.scatter(X,Y)


p= PCA(n_components=1)
p.fit(data_v)
X_p=p.transform(data_v)

print(data_v.shape)
print(X_p.shape)
print(p.components_)
print(p.explained_variance_)
print(p.mean_)

plt.scatter(p.mean_[0],p.mean_[1])
# print(data_v)

plt.plot(
    [p.mean_[0], p.mean_[0]+p.components_[0][0]*np.sqrt(p.explained_variance_[0])],
    [p.mean_[1], p.mean_[1]+p.components_[0][1]*np.sqrt(p.explained_variance_[0])]
)

plt.plot(
    [p.mean_[0], p.mean_[0]+p.components_[1][0]*np.sqrt(p.explained_variance_[1])],
    [p.mean_[1], p.mean_[1]+p.components_[1][1]*np.sqrt(p.explained_variance_[1])]
)

plt.show()

#Простота интерпретации
#Эффективность в работа с многомерными данными
#Аномальные значения в данных оказывают сильное алияение на результат












