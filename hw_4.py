#Сценарий
#Интерпритатор IPython
#Jupyter



import pandas as pd

index = [
    ('city_1', 2010),
    ('city_1', 2020),
    ('city_2', 2010),
    ('city_2', 2020),
    ('city_3', 2010),
    ('city_3', 2020),
]

population = [
    101,
    201,
    102,
    202,
    103,
    203,
]

pop = pd.Series(population, index=index)
pop_df = pd.DataFrame(
    {
        'total': pop,
        'something': [
            10,
            11,
            12,
            13,
            14,
            15,
        ]
    }
)

pop_df_1 = pop_df.loc[('city_1', 2020), 'something']
print(pop_df_1)

pop_df_2 = pop_df.loc[[('city_1', 2010), ('city_3', 2020)], ['total', 'something']]
print(pop_df_2)

pop_df_3 = pop_df.loc[[('city_1', 2010), ('city_3', 2020)], 'something']
print(pop_df_3)

pop_df_2020 = pop_df.xs(2020, level=1)
print(pop_df_2020)



index = pd.MultiIndex.from_product(
    [
        ['city_1', 'city_2'],
        [2010, 2020]
    ],
    names=['city', 'year']
)

columns = pd.MultiIndex.from_product(
    [
        ['person_1', 'person_2', 'person_3'],
        ['job_1', 'job_2']
    ],
    names=['worker', 'job']
)

data = [[i + j for j in range(6)] for i in range(4)]
df = pd.DataFrame(data, index=index, columns=columns)

df_person_1_3 = df.loc[:, (['person_1', 'person_3'], slice(None))]
print(df_person_1_3)

df_city1_person1_2 = df.loc['city_1', (['person_1', 'person_2'], slice(None))]
print(df_city1_person1_2)

idx = pd.IndexSlice
df_slice = df.loc[idx['city_1', 2010], idx['person_1', 'job_1']]
print(df_slice)

ser1 = pd.Series(['a', 'b', 'c'], index=[1, 2, 3])
ser2 = pd.Series(['b', 'c', 'f'], index=[4, 5, 6])

outer_join = pd.concat([ser1, ser2], join='outer')
print(outer_join)

inner_join = pd.concat([ser1, ser2], join='inner')
print(inner_join)










