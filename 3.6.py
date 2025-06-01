#Ансамбливые методы. В основе идея объединения нескольких переобученных (!) моделей для уменьшения эффекта переобучения
#Это называется бэггинг
#бэггинг усредняет результаты -> оптимальной классификации

#Ансамбль случайнх деревьев называется случайным лесом


#Регрессия с помощью случайных лесов



import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier

iris = sns.load_dataset('iris')


species_int = []
for r in iris.values:
    match r[4]:
        case 'setosa':
            species_int.append(1)
        case 'versicolor':
            species_int.append(2)
        case 'virginica':
            species_int.append(3)
            
species_int_df = pd.DataFrame(species_int)
# print(species_int_df.head())

data = iris[['sepal_length','petal_length']]
data['species'] = species_int

data_setosa = data[data['species'] == 1]
data_versicolor = data[data['species'] == 2]
data_virginica = data[data['species'] == 3]

# print(data.head())
# print(data.shape)

# data_setosa = data[data['species'] == 1]
# data_versicolor = data[data['species'] == 2]
# data_virginica = data[data['species'] == 3]

# data_versicolor_A = data_versicolor.iloc[:25,:]
# data_versicolor_B = data_versicolor.iloc[25:,:]

# data_virginica_A = data_versicolor.iloc[:25,:]
# data_virginica_B = data_versicolor.iloc[25:,:]


# data_df_A=pd.concat([data_virginica_A, data_versicolor_A], ignore_index = True)
# data_df_B=pd.concat([data_virginica_A, data_virginica_B], ignore_index = True)

x1_p = np.linspace(min(data['sepal_length']), max(data['sepal_length']))
x2_p = np.linspace(min(data['petal_length']), max(data['petal_length']))

X1_p, X2_p = np.meshgrid(x1_p,x2_p)

X_p = pd.DataFrame(
    np.vstack([X1_p.ravel() , X2_p.ravel()]).T, columns=['sepal_length', 'petal_length']
)
fig, ax = plt.subplots(1, 3, sharex='col', sharey='row')

ax[0].scatter(data_virginica['sepal_length'], data_virginica['petal_length'], label='virginica')
ax[0].scatter(data_setosa['sepal_length'], data_setosa['petal_length'], label='setosa')
ax[0].scatter(data_versicolor['sepal_length'], data_versicolor['petal_length'], label='versicolor')


ax[1].scatter(data_virginica['sepal_length'], data_virginica['petal_length'], label='virginica')
ax[1].scatter(data_setosa['sepal_length'], data_setosa['petal_length'], label='setosa')
ax[1].scatter(data_versicolor['sepal_length'], data_versicolor['petal_length'], label='versicolor')

ax[2].scatter(data_virginica['sepal_length'], data_virginica['petal_length'], label='virginica')
ax[2].scatter(data_setosa['sepal_length'], data_setosa['petal_length'], label='setosa')
ax[2].scatter(data_versicolor['sepal_length'], data_versicolor['petal_length'], label='versicolor')

# data_df = pd.DataFrame(data)

md = 6

X=data[['sepal_length','petal_length']]
y=data['species']

model1 = DecisionTreeClassifier(max_depth=md)
model1.fit(X,y)


y_p1 = model1.predict(X_p)

ax[0].contourf(
    X1_p,
    X2_p,
    y_p1.reshape(X1_p.shape),
    alpha=0.4,
    levels=2,
    cmap='rainbow',
    zorder=1
)

#Bagging

model2 = DecisionTreeClassifier(max_depth=md)
b = BaggingClassifier(model2,n_estimators=2,max_samples = 0.5, random_state = 1)
b.fit(X,y)

y2_p = b.predict(X_p)


ax[1].contourf(
    X1_p,
    X2_p,
    y2_p.reshape(X1_p.shape),
    alpha=0.4,
    levels=2,
    cmap='rainbow',
    zorder=1
)

#Rad=ndom Forest
model3 = RandomForestClassifier(max_depth=md)

b1 = BaggingClassifier(model2,n_estimators=2,max_samples = 0.5, random_state = 1)
b1.fit(X,y)

y3_p = b1.predict(X_p)


ax[2].contourf(
    X1_p,
    X2_p,
    y3_p.reshape(X1_p.shape),
    alpha=0.4,
    levels=2,
    cmap='rainbow',
    zorder=1
)



plt.show()

