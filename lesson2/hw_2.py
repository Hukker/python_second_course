import numpy as np

y = np.array([[1,2,3,4,5],[6,7,8,9,10]])
print(np.sum( (y>3) & (y<9) ) )

a = np.ones((3, 2))
b = np.arange(3)
b = b[:, np.newaxis]

c = a + b

print(c)