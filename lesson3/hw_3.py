import pandas as pd
import numpy as np

series1 = pd.Series([1, 2, 3, 4, 5])

series2 = pd.Series(np.array([10, 20, 30, 40, 50]))

series3 = pd.Series(7, index=['a', 'b', 'c', 'd', 'e'])

series4 = pd.Series({'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5})

print(series1)
print(series2)
print(series3)
print(series4)

df1 = pd.DataFrame({
    'col1': pd.Series([1, 2, 3]),
    'col2': pd.Series([4, 5, 6])
})

df2 = pd.DataFrame([
    {'col1': 1, 'col2': 4},
    {'col1': 2, 'col2': 5},
    {'col1': 3, 'col2': 6}
])

df3 = pd.DataFrame({
    'col1': pd.Series([1, 2, 3]),
    'col2': pd.Series([4, 5, 6])
})

df4 = pd.DataFrame(np.array([[1, 4], [2, 5], [3, 6]]), columns=['col1', 'col2'])

structured_array = np.array([(1, 4), (2, 5), (3, 6)], dtype=[('col1', 'i4'), ('col2', 'i4')])
df5 = pd.DataFrame(structured_array)

print(df1)
print(df2)
print(df3)
print(df4)
print(df5)

series_a = pd.Series([10, 20, 30], index=['a', 'b', 'c'])
series_b = pd.Series([100, 200], index=['b', 'd'])

result = series_a.add(series_b, fill_value=1)

print(result)

df = pd.DataFrame({
    'col1': [10, 20, 30],
    'col2': [40, 50, 60]
})

result = df.sub(df['col1'], axis=0)

print(result)

df = pd.DataFrame({
    'col1': [1, np.nan, 3, np.nan, 5],
    'col2': [np.nan, 2, np.nan, 4, 5]
})

ffill_df = df.ffill()

bfill_df = df.bfill()

print("ffill:\n", ffill_df)
print("bfill:\n", bfill_df)


