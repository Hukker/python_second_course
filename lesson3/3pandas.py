import numpy as np
import pandas as pd

#Pandas - расширение numpy (структурированные массивы). Строки и столбцы индексируются метками, а не только числовыми значениями

# Series, DataFrame, Index

## Series

data=pd.Series([0.25, 0.5,0.75,1.0])
#print(data)
#print(type(data))
#print(type(data.values))
#print(type(data.index))

data=pd.Series([0.25, 0.5,0.75,1.0],index=['a','b','c','d'])
#print(data['a'])

population_dict = {
    'city1':1001,
    'city2':1002,
    'city3':1003,
    'city4':1004,
    'city5':1005,

}

population = pd.Series(population_dict)
#print(population)

#print(population['city4'])
#print(population['city4':'city5'])

#Для создания Series можно использовать
# - списки Python или массивы NumPy
# - скалярные значения
# - словари
# Привести различные способы создания объектов типа Series


## DataFrame - двумерный массивы с явно определенными индексами. Последовательность "согласованных" объектов Series



population_dict = {
    'city1':1001,
    'city2':1002,
    'city3':1003,
    'city4':1004,
    'city5':1005,

}

area_dict={
    'city1':9991,
    'city2':9992,
    'city3':9993,
    'city4':9994,
    'city5':9995,
}

population = pd.Series(population_dict)
area = pd.Series(area_dict)

states = pd.DataFrame({
    'population1':population,
    'area1':area
})



#print(states)

#print(type(states.values))
#print(type(states.index))
#print(type(states.columns))

#print(states('area1'))

#DataFrame. Способы создания
# - списки словарей
# - через объекты Series
# - двумерный массив NumPy
# - словари объектов Series
# - структурированный массив нумпай

#Привести различные способы создания объектов типа DataFrame

#Incdex - способ организации ссылки на даннные объектов Series и Dataframe


#Index - это способ организации ссылки да данные объектов Series и DataFrame, Index - не изменяем упорядочен является мультимножеством(могут быть повторяющиеся значения)


ind = pd.Index([2,3,5,7,11])
#print(ind[1])
#print(ind[::2])

#ind[1] = 5


#index - следует соглашения объекта set (python)

indA = pd.Index([1,2,3,4,5])
indB = pd.Index([2,3,4,5,6])

#print(indA.intersection(indB))




#Выборка данных из Series
##как словарь

data = pd.Series([0.25,0.5,0.75,1.0],index = ['a','b','c','d'])

print('a' in data)
print(data.keys())

print(list(data.items()))
data['z']=100
data['a']=1000
print(data)


#Как одномерный массив

data = pd.Series([0.25, 0.5, 0.75, 1.0], index =['a','b','c','d'])

#print(data['a':'z'])
#print(data[0:2])
#print(data[data>0.5 & data<1])

#print(data[['a','d']])

#атрибуты-индексаторы
data = pd.Series([0.25, 0.5, 0.75, 1.0], index = [1,3,10,15])
#print(data[1])

#print(data.loc[1])
#print(data.iloc[1])



#Выборка данных из dataFrame



population_dict = {
    'city1':1001,
    'city2':1002,
    'city3':1003,
    'city4':1004,
    'city5':1005,

}

area_dict={
    'city1':9991,
    'city2':9992,
    'city3':9993,
    'city4':9994,
    'city5':9995,
}

pop = pd.Series({
    'city1':1001,
    'city2':1002,
    'city3':1003,
    'city4':1004,
    'city5':1005
}
)

area = pd.Series({
    'city1':9991,
    'city2':9992,
    'city3':9993,
    'city4':9994,
    'city5':9995,
})

data = pd.DataFrame({'area1':area,'pop1':pop,'pop':pop})
#(data)

#print(data['area1'])
#print(data.area1)

data['new'] = data['area1']
#print(data)
data['new'] = data['area1']/data['pop1']

#print(data)

data = pd.DataFrame({'area1':area,'pop1':pop})
#print(data)
#print(data.values)

#print(data.T)
#print(data['area1'])
#print(data.values[0:3])


#атрибуты-индексаторы
#print(data.iloc[:3, 1:2])

#print(data.loc[:'city4', 'pop1','pop'])



#3 Объедините 2 объектра series с одинаковыми множествами ключей так, чтобы вместо NaN было 1

#dfA = pd.DataFrame(rng.integers(0.10,(2,2)),columns=['a','b'])
#dfA = pd.DataFrame(rng.integers(0.10,(3,3)),columns=['a','b','c'])

#Pandas. 2 способа хранения отсутствуюзих значений
#Индикаторы NaN, None
#null

#None - объект(накладные расходы) не работает с sum, min

vall = np.array([1,2,3,np.nan])
#print(np.nansum(vall))


x=pd.Series(range(10),dtype=int)
#print(x)
x[0]=None
x[1]=np.nan
x1 = pd.Series(['a','b','c','d'])
#print(x1)


x2 = pd.Series([1,2,3,np.nan,None,pd.NA], dtype='Int32')
#print(x2)

#print(x2[x2.isnull()])
#print(x2[x2.notnull()])

#(x2.dropna())


df = pd.DataFrame(
    [
        [1,2,3,np.nan,None,pd.NA],
        [1,2,3,4,5,6],
        [1,np.nan,3,4,np.nan,6]
    ])
print(df)
print(df.dropna())

#how
# - all все NA
# - any хотя бы одно
# - thresh = x, минимум x непустых









