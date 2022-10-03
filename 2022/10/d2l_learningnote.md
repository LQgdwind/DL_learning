# d2l learning note

## Chapter 1 绪论
略

## Chapter 2 预备知识

### Chapter 2.1 numpy与pytorch
pytorch的基础语法与numpy的语法十分的类似。

可以使用torch.from_numpy以及具体的张量对象的numpy方法进行相互转换。


```python3
import numpy as np
import torch 
a = np.array([1,2,3]).astype(np.float64)
b = torch.from_numpy(a)
c = b.numpy()
print(a)
print(b)
print(c)
```
要将大小为1的张量对象转换为python标量，可以使用item方法

```python3
import torch
a = torch.tensor([1.356],dtype=torch.float64)
b = a.item()
print(a)
print(b)
```

### Chapter 2.2 pandas与pytorch

```python3
import os
import pandas as pd
data_file=os.path.join('data.csv')
with open(data_file,'w+') as f:
    f.write('NumRooms,Alley,Price\n')
    # 列名
    f.write('NA,Pave,127500\n')
    # 每⾏表⽰⼀个数据样本
    f.write('2,NA,106000\n')
    f.write('4,NA,178100\n')
    f.write('NA,NA,140000\n')

data = pd.read_csv("data.csv")
print(data)
```

为了处理缺失的数据，典型的⽅法包括插值法和删除法，其中插值法⽤⼀个替代值弥补缺失值，⽽删除法则直接忽略缺失值。在这⾥，我们将考虑插值法。

通过位置索引iloc，我们将data分成inputs和outputs，其中前者为data的前两列，⽽后者为data的最后⼀列。对于inputs中缺少的数值，我们⽤同⼀列的均值替换“NaN”项。

```python3
import torch
import pandas as pd
inputs,outputs = data.iloc[:,0:2],data.iloc[:,-1]
inputs = inputs.fillna(inputs.mean(axis=0))
inputs = pd.get_dummies(inputs,dummy_na=True)

# 可以通过将values属性作为Torch.tensor的参数构建tensor
inputs = torch.tensor(inputs.values)
outputs = torch.tensor(outputs.values)
print(inputs)
print(outputs)
```

### Chapter 2.3 线性代数

用法与numpy相似

注意一些细节

1.比如在使用sum方法的时候，会对相应的axis进行降维

如果想要tensor保持原来的形状，需要加上keepdim = True

比如 a = y.sum(axis = 2,keepdim=True)

2.矩阵的L1范数即为绝对值，通常使用abs函数调用计算

比如a = y.abs().sum()

3.矩阵的Frobenius范数（Frobenius norm）是矩阵元素平⽅和的平⽅根

可以使用norm方法调用计算

比如a = torch.norm(y)

结果会是一个元素的张量。

如果想要得到python标量，可以调用Item()函数

比如a = torch.norm(y).item()

### Chapter 2.4 微积分

略

### Chapter 2.5 自动求导

我们可以使用detach函数来分离梯度

```python3
import torch
x = torch.tensor([1.,2.,3.])
x.requires_grad = True
y = x**2
w = y.detach()
z = w * x
x.grad.zero_()
z.backward(torch.ones_like(x))
print(x.grad == w)
```

### Chapter 2.6 概率论

略

### Chapter 2.7 查阅文档

略

## Chapter 3 线性神经网络

### Chapter 3.1 线性回归

线性回 归基于⼏个简单的假设：

⾸先，假设⾃变量x和因变量y之间的关系是线性的，即y可以表⽰为x中元素的加权
和，这⾥通常允许包含观测值的⼀些噪声；

其次，我们假设任何噪声都⽐较正常，如噪声遵循正态分布。

为了解释线性回归，我们举⼀个实际的例⼦：我们希望根据房屋的⾯积（平⽅英尺）和房龄（年）来估算房
屋价格（美元）。为了开发⼀个能预测房价的模型，我们需要收集⼀个真实的数据集。这个数据集包括了房
屋的销售价格、⾯积和房龄。

在机器学习的术语中，该数据集称为训练数据集（training data set）或训练集
（training set）。

每⾏数据（⽐如⼀次房屋交易相对应的数据）称为样本（sample），也可以称为数据点（data
point）或数据样本（data instance）。

我们把试图预测的⽬标（⽐如预测房屋价格）称为标签（label）或⽬
标（target）。

预测所依据的⾃变量（⾯积和房龄）称为特征（feature）或协变量（covariate）。

### Chapter 3.2 线性回归的从零开始实现

