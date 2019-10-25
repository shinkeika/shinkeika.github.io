---
title: 'Numpy pandas matplotlib学习笔记'
date: 2019-10-25
permalink: /posts/2019/10/blog-post-6/
tags:
  - numpy
  - pandas
  - matplotlib
---

Numpy pandas matplotlib学习笔记
虽然简单，但是还是要全部过一遍

[TOC]

### numpy


```python
import numpy as np
```

#### 把列表转换成矩阵的方法 [][]是列表


```python
array =np.array([[1,2,3],[2,3,4]])  # 把列表转换成矩阵的方法 [][]是列表
print(array)
```

#### 查看维度


```python
print('number of dim:',array.ndim)  # 查看维度
```

    number of dim: 2


#### 查看形状


```python
print('shape:',array.shape)  # 查看形状
```

    shape: (2, 3)


#### 查看一共有多少元素


```python
print('size:',array.size)  # 查看一共有多少元素
```

    size: 6



```python

```

#### 定义type  int32 float32 float64


```python
a = np.array([[1,2,3],
               [2,3,4]
             ],
             dtype=np.int)  # 定义type  int32 float32 float64

```


```python
print(a.dtype)
```

    int64


#### 生成全部为0的矩阵


```python
a = np.zeros((3,4))  # 生成全部为0的矩阵
```


```python
a
```




    array([[0., 0., 0., 0.],
           [0., 0., 0., 0.],
           [0., 0., 0., 0.]])



#### 全部为1的矩阵


```python
a = np.ones((3,4), dtype=np.int16)  # 全部为1的矩阵
```


```python
a
```




    array([[1, 1, 1, 1],
           [1, 1, 1, 1],
           [1, 1, 1, 1]], dtype=int16)



#### 全部为空的


```python
a = np.empty((3,4))  # 全部为空的
```


```python
a
```




    array([[0., 0., 0., 0.],
           [0., 0., 0., 0.],
           [0., 0., 0., 0.]])



#### 生成有序的矩阵 起始值，终止值，步长


```python
a = np.arange(10,20,2)  # 生成有序的矩阵 起始值，终止值，步长
```


```python
a
```




    array([10, 12, 14, 16, 18])



#### 生成有序的三行四列


```python
a = np.arange(12).reshape((3,4))  # 生成有序的三行四列
```


```python
a
```




    array([[ 0,  1,  2,  3],
           [ 4,  5,  6,  7],
           [ 8,  9, 10, 11]])



#### 生成线段


```python
a = np.linspace(1, 10, 5)  # 生成线段
```


```python
a
```




    array([ 1.  ,  3.25,  5.5 ,  7.75, 10.  ])



#### 改形状，但是注意6 = 2*3


```python
a = np.linspace(1, 10, 6).reshape((2,3))  #改形状，但是注意6 = 2*3
```


```python
a
```




    array([[ 1. ,  2.8,  4.6],
           [ 6.4,  8.2, 10. ]])




```python
a = np.array([10,20,30,40])
b = np.arange(4)
```

#### 减法


```python
c = a - b  # 减法
```


```python
c
```




    array([10, 19, 28, 37])



#### 加法


```python
c = a + b  # 加法
```


```python
c 
```




    array([10, 21, 32, 43])



#### 乘法


```python
c = a * b  # 乘法
```


```python
c
```




    array([  0,  20,  60, 120])



#### 平方 


```python
c = b ** 2  # 平方 
```


```python
c
```




    array([0, 1, 4, 9])



#### 求三角函数


```python
c = 10 * np.sin(a)  # 求三角函数
```


```python
c 
```




    array([-5.44021111,  9.12945251, -9.88031624,  7.4511316 ])



#### 找出矩阵中小于某值的数


```python
print(b<3)  # 找出矩阵中小于某值的数
```

    [ True  True  True False]


#### 矩阵乘法


```python
a = np.array([[1,1],[0,1]])
b = np.arange(4).reshape(2,2)
```


```python
a
```




    array([[1, 1],
           [0, 1]])




```python
b
```




    array([[0, 1],
           [2, 3]])




```python
#### 逐个相乘
```


```python
c = a * b
c
```




    array([[0, 1],
           [0, 3]])



#### 矩阵乘法(两种写法)


```python
c = np.dot(a,b)
c
```




    array([[2, 4],
           [2, 3]])




```python
c_1 = a.dot(b)
c_1
```




    array([[2, 4],
           [2, 3]])



#### 随机产生


```python
a = np.random.random((2,4))
a
```




    array([[0.48345384, 0.78123423, 0.73138538, 0.84352961],
           [0.73087187, 0.82949178, 0.422308  , 0.53558165]])



#### 最大，最小，求和


```python
np.sum(a)
np.min(a)
np.max(a)
```




    0.8435296071382901



#### 求和的维度 axis=0 列  axis = 1 行


```python
np.sum(a, axis=0)
```




    array([1.21432571, 1.61072601, 1.15369338, 1.37911125])



#### 


```python
A = np.arange(2, 14).reshape((3,4))
```


```python
A
```




    array([[ 2,  3,  4,  5],
           [ 6,  7,  8,  9],
           [10, 11, 12, 13]])



#### 一些小函数


```python
print(np.argmin(A))
```

    0



```python
np.argmax(A)
np.mean(A)  # A.mean() 也可以
np.average(A)  # A.average不行
np.median(A)  # 中位数
np.cumsum(A)  # 累加
np.diff(A)  # 差
np.nonzero(A)  # 非0的数  输出对应每个位置 一个array行 一个array列
np.sort(A)  # 排序
np.transpose(A)  # 矩阵反向,转置  A.T
np.clip(A, 5, 9)  # 小于5的=5，大于9的变成9
A.flatten()  # 放平
A.flat  # 迭代器
```




    <numpy.flatiter at 0x7fe5de12f000>



#### 根据索引找


```python
A = np.arange(3,15).reshape(3,4)
A
```




    array([[ 3,  4,  5,  6],
           [ 7,  8,  9, 10],
           [11, 12, 13, 14]])




```python
print(A[2])  # 索引为2
```

    [11 12 13 14]



```python
print(A[2,2])  # 索引为2,2
```

    13



```python
print(A[2,:])  # 行索引为2的所有
```

    [11 12 13 14]



```python
for row in A:  # 迭代行
    print(row)
```

    [3 4 5 6]
    [ 7  8  9 10]
    [11 12 13 14]



```python
for column in A.T:  # 迭代列  trick的方法
    print(column)
```

    [ 3  7 11]
    [ 4  8 12]
    [ 5  9 13]
    [ 6 10 14]



```python
for item in A.flat:  # 迭代项目
    print(item)
```

    3
    4
    5
    6
    7
    8
    9
    10
    11
    12
    13
    14



```python
print(A.flatten())
```

    [ 3  4  5  6  7  8  9 10 11 12 13 14]


#### 合并


```python
A = np.array([1,1,1])
B = np.array([2,2,2])
```

#### 上下合并


```python
print(np.vstack((A,B)))  # vertica stack 注意传tuple 可以合并多个
```

    [[1 1 1]
     [2 2 2]]


#### 左右合并


```python
print(np.hstack((A,B))) 
```

    [1 1 1 2 2 2]



```python

```

#### 加维度


```python
print(A[np.newaxis,:].shape)
C = A[:,np.newaxis]
```

    (1, 3)



```python
print(A[:,np.newaxis].shape)
D = A[:,np.newaxis]
```

    (3, 1)


#### 多个矩阵合并


```python
print(np.concatenate((C,D,C,D), axis=1))
```

    [[1 1 1 1]
     [1 1 1 1]
     [1 1 1 1]]


#### 分割数组


```python
A = np.arange(12).reshape((3,4))
```


```python
print(A)
```

    [[ 0  1  2  3]
     [ 4  5  6  7]
     [ 8  9 10 11]]



```python
print(np.split(A, 2, axis=1))  # 从行进行分割
```

    [array([[0, 1],
           [4, 5],
           [8, 9]]), array([[ 2,  3],
           [ 6,  7],
           [10, 11]])]



```python
print(np.split(A, 3,axis=0))  # 从列进行分割
```

    [array([[0, 1, 2, 3]]), array([[4, 5, 6, 7]]), array([[ 8,  9, 10, 11]])]


#### 不等量分割


```python
print(np.array_split(A,3,axis=1))
```

    [array([[0, 1],
           [4, 5],
           [8, 9]]), array([[ 2],
           [ 6],
           [10]]), array([[ 3],
           [ 7],
           [11]])]



```python
print(np.vsplit(A,3))  # 纵向分割
```

    [array([[0, 1, 2, 3]]), array([[4, 5, 6, 7]]), array([[ 8,  9, 10, 11]])]



```python
print(np.hsplit(A,2))  # 横向分割
```

    [array([[0, 1],
           [4, 5],
           [8, 9]]), array([[ 2,  3],
           [ 6,  7],
           [10, 11]])]



```python

```


```python

```

####  赋值


```python
a = np.arange(4)
```


```python
a
```




    array([0, 1, 2, 3])




```python
b = a  # 赋值
c = a
d = b
```


```python
a[0] = 11   # 改变a中的值
```


```python
b  # 全变了  是个引用改变
```




    array([11,  1,  2,  3])




```python
b is a  # b 就是 a
```




    True




```python
d is a  # d也是a
```




    True




```python
d[1:3] = [22, 33]
```


```python
a  # a也改变了
```




    array([11, 22, 33,  3])



#### 赋值但是不关联


```python
b = a.copy()  # deep copy
```


```python
b
```




    array([11, 22, 33,  3])




```python
a[3] = 44
```


```python
b  # 没有改变
```




    array([11, 22, 33,  3])




```python

```

