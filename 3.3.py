#Задача: на основе наблюдаемых точек пострить прямую, которая отображает связь между двумя или более переменными .abs
#регрессия пытается подогнать функцию к наблюдаемым данным , чтобы спрогнозировать данные
#Линейная регрессия подгоняем данные к прямой линии, пытаемся установить линейную связь между переменными и предсказать новые данные
import random
import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from numpy.linalg import inv, qr


# features, target = make_regression(n_samples=100, n_features=1,n_informative=1,n_targets=1, noise=15,random_state=1)

# print(features)
# print(target)

# model = LinearRegression().fit(features, target)

# plt.scatter(features, target, s =0.2)


# x = np.linspace(features.min(), features.max(), 100)

# #y = kx +b 
# plt.plot(x , model.coef_[0]*x + model.intercept_, c='red' )

# plt.show()


# ## Простая линейная регрессия
# # Линейная зависимость
# # + прогноз на всех новых данных
# # + анализ взаимного влияния переменных друг на друга

# # - точки обучаемых данных не будут точно жедать на прямой
# # - не позволяет делать прогнозы вне диапозона данных

#данные, на основании которых разрабатывается модель - это выборка из совокупности. Хотелось бы чтобы это была репрезентативная выборка


# data = np.array(
#     [
#         [1,5],
#         [2,7],
#         [3,7],
#         [4,10],
#         [5,11],
#         [6,14],
#         [7,17],
#         [8,19],
#         [9,22],
#         [10,28],
        
#     ]
# )

# x = data[:,0]
# y=data[:,1]
# n= len(x)

# w_1 = (
#     n*sum(x[i]*y[i] for i in range(n)) - sum(x[i] for i in range(n))*sum(y[i] for i in range(n)) 
#     ) / (n * sum(x[i]**2 for i in range(n)) - sum(x[i] for i in range(n))**2)


# w_0 = sum(y[i] for i in range(n))/n - w_1 * (sum(x[i] for i in range(n)))/n

# print(w_1, w_0)




# #Метод обратных матриц

# # 
# #
# #  X = [1 x1]    y = [y1]
# #      [1 x2]        ...
# #      ...           [yn]
# #      [1 xn]
# #
# # w = (X^T*X)**(-1)*x^T*y
# #
# #

# x_1 = np.vstack([x, np.ones(len(x))]).T
# w = inv(x_1.transpose() @ x_1) @ (x_1.transpose() @ y)

# print(w)


# # Разложенеие матриц X = QR -> w = E**(-1)Q^Ty
# #QR раложение
# # Минимизирует ошибку вычисления


# Q,R = qr(x_1)

# w = inv(R).dot(Q.transpose()).dot(y)
# print(w)
# # Градиентный спуск

def f(x):
    return (x-3)**2 + 4
ax = plt.gca()
ax.xaxis.set_major_locator(plt.MultipleLocator(0.5))

x=np.linspace(-10,10,100)
plt.plot(x,f(x))
plt.grid()
plt.show()

def dx_f(x):
    return 2*x-6

L=0.001

iterations = 100_000

x - random.randint(0,5)

for i in range(iterations):
    d_x = dx_f(x)
    x-= L*d_x
    
print(x, f(x))

#смещение модели означает, что предпочтение отдается определеной схеме, а не кривых со сложной структурой
#если в модель добавить смещение то есть риск недообучения
#задача сводится к минимизации функции потерь и некоторого смещения


#-ridge регрессия(гребневая)
#добавляет смещение в виде штрафа(хуже подгонка под имющиеся данные)
#-lasso регрессия
#удаление некоторых признаков

#механически применить регрессию к данным(сделать на основе полученной модели прогноз, после этого считать что все в порядке)

# import numpy as np
# import  pandas as pd
# import random
# from sklearn.linear_model import LinearRegression
# import matplotlib.pyplot as plt
# from sklearn.model_selection import KFold, cross_val_score, train_test_split


data = np.array(
    [
        [1,5],
        [2,7],
        [3,7],
        [4,10],
        [5,11],
        [6,14],
        [7,17],
        [8,19],
        [9,22],
        [10,28]
    ]
)

x = data[:,0]
y = data[:,1]

n = len(x)

w1 = 0.0
w0 = 0.0

L = 0.001

iterations = 100000

sample_size = 1

for i in range(iterations):
    idx = np.random.choice(n,sample_size, replace= False)
    D_w0 = 2 * sum(-y[idx]+w0+w1*x[idx])
    D_w1 = 2 * sum((x[idx] * (-y[idx] +w0+w1*x[idx])))
    w1 -= L*D_w1
    w0 -= L*D_w0
print(data)
print(w1,w0)
#Градиентный спуск - пакетный градиентный спуск. Для работы используются все доступны обучающие ланные
#Стохастический градиентный спуск - на каждой итерации обучемся только на одной выборке из данных
# - сокращение число вычислений
# - вносим смещение -> боремся с переобучением
# Мини-пакетный градиентный спуск, на каждой итерации используется несколько выборок


#как оценить насколько сильно промахиваются прогнозы при использовании линейной регрессии
#для оценки степени взаимосвязи можно

data_df = pd.DataFrame(data)
# print(data_df.corr(method = 'pearson'))
# data_df[1] = data_df[1].values[::-1]
# print(data_df.corr(method = 'pearson'))

#коэффициент корреляции помогает понять есть ли связь между двумя пременными
#Обучающие и тестовые выборки
#Основной метод борьбы с переобучением, заключается в том что набор данных делитя на обучющие и тестовые выборкт

X = data_df.values[:,:-1]
Y = data_df.values[:,-1]


X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=1/3)

print(X_train, Y_train)
print(X_test,Y_test)

model = LinearRegression()
model.fit(X_train, Y_train)

r = model.score(X_test, Y_test)
print(r) 


kfold = KFold(n_splits =3, random_state =1,shuffle=True) #трехкратная перекрестная валидация
results = cross_val_score(model,X,Y,cv=kfold)

print(results)
print(results.mean(), results.std())

#валидационная выборка - для сравнения различных моделей или конфигурации