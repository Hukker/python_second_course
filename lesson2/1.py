import numpy as np
import sys
import array
'''
x = 1
print(type(x))
print(sys.getsizeof(x))

x = 'hello'
print(type(x))
print(sys.getsizeof(x))

l1 = list([])
print(sys.getsizeof(l1))

l2 = list([1,2,3])

l3 = list([1, "2", True])

print(l3)
print(sys.getsizeof((l3)))

a1 = array.array('i',[1,2,3])
print(sys.getsizeof(a1))
print(type(a1)

a = np.array([1,2,3,4,5])
print(type(a), a)


#повышающее приведение типов
a = np.array([1.23,2,3,4,5], dtype=int)
print(type(a), a)

a = np.array([range(i,i+3) for i in [2,4,6]])
print(a, type(a))

a = np.zeros(10,dtype = int)
print(a, type(a))

print(np.ones((3,5), dtype=float))

print(np.full((4,5), 3.1415))

print(np.arange(0,20, 2))


print(np.eye(4))
'''

#МАССИВЫ

np.random.seed(1)
x1 = np.random.randint(10, size=3)
x2 = np.random.randint(10,size=(3,2))
x3 = np.random.randint(10,size=(3,2,1))
#print(x1)
#print(x2)
#print(x3)

#print(x1.ndim, x1.shape, x1.size)
#print(x2.ndim, x2.shape, x2.size)
#print(x3.ndim, x3.shape, x3.size)

a = np.array([1,2,3,4,5])
#print(a[0])
#print(a[-2])


a= np.array([[1,2], [3,4]])
#print(a)
#print(a[-1,-1])

a = np.array([1,2,3,4,7,2,9,53,2,5,2])
b = np.array([1.0,2,3,4])

print(a)
print(b)

a[0] = 10.1
print(a)

##СРЕЗЫ

print(a[:3:])
print(a[::-1])

a = np.arange(1,13)
print(a)

print(a.reshape(3,4))

x = np.array([1,2,3])
y = np.array([4,5])
z = np.array([6])
v = np.concatenate([x,y,z])
print(v)


x = np.array([1,2,3])
y = np.array([4,5,6])
r1 = np.vstack([x,y])
#print(r1)
#print(np.hstack([r1,r1]))

#ВЕКТРОИЗИРОВАННЫЕ ОПЕРАЦИИ- независимо к каждому элекменту массива
x = np.arange(10)
#print(x)

#print(x*2+1)

#УНИВЕРСАЛЬНЫЕ ФУНКЦИИ
#print(np.add(np.multiply(x,2),1))



