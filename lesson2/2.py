import numpy as np

rng = np.random.default_rng(1)
s = rng.random(50)


#print(s)
#print(sum(s))
#print(np.sum(s))

a = np.array([
    [1,2,3,4,5],
    [6,7,8,9,10]
])
#print(np.sum(a))
#print(np.sum(a,axis=0))
#print(np.sum(a,axis=1))


#NaN = Not a number
#print(np.nanmin(a))
#print(np.nanmin(a, axis=1))

#Транслирование (broadcasting)
#набор правил, которые позволяют осуществлять бинарные операции с массивами разных форм и размеров
a = np.array([0,1,2])
b = np.array([0,1,2])
#print(a+b)
#print(a+7)#транслируется в [7,7,7], подстраивается под размер массива a
a = np.array([
    [0,1,2],
    [3,4,5]
])
#print(a+5)

a=np.array([0,1,2])
b=np.array([[0],[1],[2]])
#print(a+b)

#Правило
#1.Если размерности массивов отличаются, то форма массива, то форма массива дополняется 1 с меньшей стороны
#2.Если формы массивов не совпадают в какой-то размерности, то если у массива форма равна 1, то он растягивается до соответствия формы
#3.Если после применения этих правил в каком-то измерении и ни один из них не равен 1, то генерируется ошибка




#a = np.array([0,1,2])
#b = np.array([5])
#print(a.ndim)
#print(a+b)

#a         (2,3)
#b (1,) -> (1,1) -> (2,3)
#print(a + b)

a = np.ones((2,3))
b = np.arange(3)

#print(a)
#print(b)

#print(a.ndim, a.shape)
#print(b.ndim, b.shape)

#(2,3)    (2,3)    (2,3)
#(3, ) -> (1,3) -> (2,3)

c = a + b

# [ 0 0 0 ]  [ 0 1 2 ]
# [ 1 1 1 ] +[ 0 1 2 ]
# [ 2 2 2 ]  [ 0 1 2 ]

#print(c, c.shape)

a = np. ones((3,2))
b = np.arange(3)

#print(b)
# 2 (3,2) -> (3,2)    (3,2)
# 1 (3, ) -> (1,3) -> (3,3)
#c=a+b

#Q1 Что надо изменить, чтобы последний пример заработал

X = np.array([
    [1,2,3,4,5,6,7,8,9],
    [9,8,7,6,5,4,3,2,1]
])

Xmean0 =X.mean(0)
#print(Xmean0)
Xcenter0 = X- Xmean0
#print(Xcenter0)

Xmean1 = X.mean(1)
#print(Xmean1)
Xmean1 = Xmean1[:,np.newaxis]

Xcenter1 = X - Xmean1
#print(Xcenter1)

x=np.linspace(0,5,50)
y=np.linspace(0,5,50)[:, np.newaxis]

z=np.sin(x)**3 + np.cos(20+y*x)*np.sin(y)

import matplotlib.pyplot as plt
#print(z.shape)
#plt.imshow(z)
#plt.colorbar()
#plt.show()

x = np.array([1,2,3,4,5])
#print(x < 3)
#print(np.less(x,3))

#print(np.sum(x<4))

# Q2 пример для у. Вычислить количество элементов(по обоим размерностям), значения которых больше 3 и меньше 9

x = np.array([1,2,3,4,5])
y = print(x[x<3])

#print(bin(42))
#print(bin(59))
#print(bin(42 & 59))


#Векторизация индекса

x= np.array([1,2,3,4,5,6,7,8,9])
index = [1,5,7]

#print(x[index])
index = [[1,5,7],[2,4,8]]
#print(x[index])
#Форма результата отражает форму массива индексов
x = np.arange(12).reshape((3,4))
#print(x)
#print(x[2])
#print(x[2,[2,0,1]])
#print(x[1:, [2,0,1]])

x=np.arange(10)
i=np.array([2,1,8,4])
#print(x)
x[i]=999

#print(x)



#СОРТИРОВКА

x= [2,54,9,190,3,86,321,67,4]
print(sorted(x))
print(np.sort(x))


#СТРУКТУРИРОВАННЫЕ МАССИВЫ
data=np.zeros(4, dtype = {
    'names':(
        'name',
        'age'
    ),
    'formats':(
        'U10', 'i4'
    )
})

print(data.dtype)
name = [ 'name1','name2','name3','name4']
age = [10,20,30,40]
data['name'] = name
data['age'] = age


print(data)

print(data[data['age']>20]['name'])


#МАССИВЫ ЗАПИСЕЙ

data_rec = data.view(np.recarray)

print(data_rec)
print(data_rec[0])
print(data_rec[-1].name)
