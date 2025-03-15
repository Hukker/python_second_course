import numpy as np
import pandas as pd

index = [
    ('city1' , 2010),
    ('city2' , 2020),
    ('city3' , 2010),
    ('city4' , 2020),
    ('city5' , 2010),
    ('city6' , 2020),
]

population = [
    101,
    201,
    102,
    301,
    203,
    103
]

pop = pd.Series(population, index= index)
#print(pop[ [i for i in pop.index if i[1]==2020]])

#MultiIndex


index = pd.MultiIndex.from_tuples(index)


pop = pop.reindex(index)
#print(pop[:, 2020])

pop_df = pop.unstack()
#print(pop_df)

#print(pop_df.stack(()))

index = [
    ('city1', 2010, 1),
    ('city1', 2010, 2),

    ('city1', 2020, 1),
    ('city1', 2020, 2),

    ('city2', 2010, 1),
    ('city2', 2010, 2),

    ('city2', 2020, 1),
    ('city2', 2020, 2),

    ('city3', 2010, 1),
    ('city3', 2010, 2),

    ('city3', 2020, 1),
    ('city3', 2020, 2),

]

population = [
    101,
    201,
    102,
    2010,
    102,
    103,
    301,
    201,
    532,
    101,
    890,
    1
]

pop = pd.Series(population, index=index)

#print(pop)






index = pd.MultiIndex.from_tuples(index)
pop = pop.reindex(index)
#print(pop)
#print(pop[:,2010])
#print(pop[:,:,2])


pop_df = pop.unstack()
#print(pop_df)


pop_df = pd.DataFrame(
    {
        'total':pop,
        'something':[
            101,
            201,
            102,
            2010,
            102,
            103,
            301,
            201,
            532,
            101,
            890,
            1

        ]
    }
)
#print(pop_df)
#print(pop_df['something'])
#pop_df_1 = pop_df.loc['city1','something']
#print(pop_df_1)

#разоьраться как используются мультииндексные ключи в данном примере

##Как можно создавать мультииндекс
## - список массиов, задающих значение индекса на каждом уровне
## - список картежей, задающих значение индекса в каждой точке
## - декартово произведение обычных индексов
## - описание внутреннеого представления : levels, codes
i1 = pd.MultiIndex.from_arrays(
    (
        ['a','a','b','b'],
        [1,2,1,2]
    )
)
i2 = pd.MultiIndex.from_tuples(
    [
        ('a' , 1),
        ('a' , 2),
        ('b' , 1),
        ('b' , 2)

    ]
)

i3 = pd.MultiIndex.from_product(
    [
        ['a','b'],
        [1,2]
    ]
)
#print(i1)
#print(i2)
#print(i3)

i4 = pd.MultiIndex(
    levels = [
        ['a','b','c'],
        [1,2]
    ],
    codes = [
        [0,0,1,1,2,2], # a a b b
        [0,1,0,1,0,1] # 1 2 1 2
    ]
)

#print(i4)

data = {
    ('city1', 2010): 100,
    ('city2', 2020): 200,
    ('city3', 2010): 1001,
    ('city4', 2020): 2001,
}
s = pd.Series(data)
#print(s)
s.index.names = ['city','year']
#print(s)

index = pd.MultiIndex.from_product(
    [
        ['city1','city2'],
        [2010,2020]
    ],
    names=['city','year']
)
#print(index)

columns = pd.MultiIndex.from_product(
    [
        ['person1','person2','person3'],
        ['job1','job2']
    ],
    names=['worker','job']
)
rng = np.random.default_rng(1)
data = rng.random((4,6))
#print(data)
data_df = pd.DataFrame(data,index=index,columns = columns)
#print(data_df)

# 2. Из получившихся данных выбрать данные по
# - 2020 году для всех столбцов
# - job1 для всех строк
# - для city1 и job2

#ИНДЕКСАЦИЯ И СРЕЗЫ ПО МУЛЬТИИНДЕКСУ

data = {
    ('city1', 2010): 100,
    ('city1', 2020): 200,
    ('city2', 2010): 1001,
    ('city2', 2020): 2001,
    ('city3', 2010): 20001,
    ('city3', 2020): 20001,
}

s = pd.Series(data)
s.index.names = ['city', 'year']


#print(s['city1', 2010])
#print(s['city1'])
#print(s.loc['city1':'city2'])
#print(s[s > 2000])
#print(s[['city1','city3']])

#Выполнить запрос на получение следующих данных
# - все данные по person1 и person3
# - все данные по первому городу и первым двум персонам(использовать срезы)
# приведите пример с использованием pd.IndexSlice



# Перегруппировка мультииндексов
index = pd.MultiIndex.from_product(
    [
        ['a','c','b'],
        [1,2]
    ]
)

data = pd.Series(rng.random(6), index = index)
data.index.names = ['char','int']

#print(data)
data=data.sort_index()

#print(data['a':'b'])

ser1 = pd.Series(['a','b','c'], index=[1,2,3])
ser2 = pd.Series(['d','e','f'], index=[4,5,6])
#print(pd.concat([ser1,ser2],verify_integrity=False))
#print(pd.concat([ser1,ser2],verify_integrity=True))
#print(pd.concat([ser1,ser2],ignore_index=True))

print(pd.concat([ser1,ser2],keys=['x','y']))
print(pd.concat([ser1,ser2], join='outer'))
print(pd.concat([ser1,ser2], join='inner'))

#4 Привести пример использования inner и outer джойнов для Series (на данных преыдущего примера)






