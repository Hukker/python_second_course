#Наивный байес
#Набор моделей, которые предлагают быстрые и простые алгоритмы классификации
#
#
#
#
#
#
#Гауссовский наивный байесовский классификатор
#Допущение состоит в том, что данные всех категорий взяты из простого нормального распределения


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.naive_bayes import GaussianNB


iris = sns.load_dataset('iris')
# print(iris.head())

# sns.pairplot(iris, hue='species')

data = iris[['sepal_length', 'petal_length', 'species']]
# print(data.head())
#setosa versicolor 
#setosa virginica

data_df = data[(data['species'] == 'setosa' )| (data['species'] == 'versicolor')]


X = data_df[['sepal_length', 'petal_length']]
y = data_df['species']
model = GaussianNB(X,y)

model.fit(X,y)

data_df_seposa = data_df[data_df['species'] == 'setosa']
data_df_versicolor = data_df[data_df['species']=='versicolor']

plt.scatter(data_df_seposa['sepal_length'], data_df_seposa['petal_length'])
plt.scatter(data_df_versicolor['sepal_length'], data_df_versicolor['petal_length'])

x1_p = np.linspace(min(data_df['sepal_lrngth']), max(data_df['sepal_length']), 100)
x2_p = np.linspace(min(data_df['petal_lrngth']), max(data_df['petal_length']), 100)

X1_p, X2_p = np.meshgrid(x1_p, x2_p)

X_p = pd.DataFrame(
    np.vstack([X1_p.ravel(), X2_p.ravel()]).T
)
model.predict(X_p)




print(data_df.shape)
plt.show()










