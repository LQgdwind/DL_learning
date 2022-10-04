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

1. <b>生成数据集</b>

    为了简单起⻅，我们将根据带有噪声的线性模型构造⼀个⼈造数据集。

    我们的任务是使⽤这个有限样本的数据集来恢复这个模型的参数。

    我们将使⽤低维数据，这样可以很容易地将其可视化。

    在下⾯的代码中，我们⽣成⼀个包含1000个样本的数据集，每个样本包含从标准正态分布中采样的2个特征。

    我们使⽤线性模型参数w = [2, −3.4]^⊤、b = 4.2 和噪声项ϵ⽣成数据集及其标签：

   y = Xw + b + ϵ.

```python3
def synthetic_data(w, b, num_examples):
    """生成y = Xw+b+噪声"""
    x = torch.normal(0, 1, (num_examples,len(w)))
    y = torch.matmul(x, w) + b
    y += torch.normal(0, 0.01, y.shape)
    print(y.shape)
    return x,y.reshape((-1, 1))
```
2. <b>读取数据集</b>

   回想⼀下，训练模型时要对数据集进⾏遍历，每次抽取⼀⼩批量样本，并使⽤它们来更新我们的模型。由于
   这个过程是训练机器学习算法的基础， 
   所以有必要定义⼀个函数，该函数能打乱数据集中的样本并以⼩批量
   ⽅式获取数据。
   在下⾯的代码中，我们定义⼀个data_iter函数，该函数接收批量⼤⼩、特征矩阵和标签向量作为输⼊，⽣
    成⼤⼩为batch_size的⼩批量。每个⼩批量包含⼀组特征和标签。
   ```python3
        def data_iter(batch_size, features, labels):
            num_examples = len(features)
            indices = list(range(num_examples))
            random.shuffle(indices)
            for i in range(0,num_examples,batch_size):
                batch_indices = torch.tensor((indices[i:min(i+batch_size,num_examples)]))
                yield features[batch_indices],labels[batch_indices]
   ```

3. <b>初始化模型参数</b>

   在我们开始⽤⼩批量随机梯度下降优化我们的模型参数之前，我们需要先有⼀些参数。在下⾯的代码中，我
   们通过从均值为0、标准差为0.01的正态分布中采样随机数来初始化权重，并将偏置初始化为0。

4. <b> 定义模型</b>

   接下来，我们必须定义模型，将模型的输⼊和参数同模型的输出关联起来。回想⼀下，要计算线性模型的输出，我们只需计算输⼊特征**X**和模型权重**w**的矩阵-向量乘法后加上偏置*b*。注意，上⾯的**Xw**是⼀个向量，⽽*b*是⼀个标量。回想⼀下广播机制：当我们⽤⼀个向量加⼀个标量时，标量会被加到向量的每个分量上。

   ```python3
   def linreg(X, w, b): 
   """线性回归模型"""
   return torch.matmul(X, w) + b
   ```

5. <b>定义损失函数</b>

   因为需要计算损失函数的梯度，所以我们应该先定义损失函数。这⾥我们使⽤平⽅损失函数。

   在实现中，我们需要将真实值y的形状转换为和预测值y_hat的形状相同。

   ```python3
   def squared_loss(y_hat, y): 
   """均⽅损失"""
   return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2
   ```

6. <b>定义优化算法</b>

   在每⼀步中，使⽤从数据集中随机抽取的⼀个⼩批量，然后根据参数计算损失的梯度。接下来，朝着减少损失的⽅向更新我们的参数。下⾯的函数实现⼩批量随机梯度下降更新。该函数接受模型参数集合、学习速率和批量大小作为输⼊。每⼀步更新的⼤⼩由学习速率lr决定。因为我们计算的损失是⼀个批量样本的总和，所以我们⽤批量⼤⼩（batch_size）来规范化步⻓，这样步⻓⼤⼩就不会取决于我们对批量⼤⼩的选择。

   ```python3
   def sgd(params, lr, batch_size): #@save
   """⼩批量随机梯度下降"""
   with torch.no_grad():
   for param in params:
   param -= lr * param.grad / batch_size
   param.grad.zero_()
   ```

7. <b>训练</b>

   现在我们已经准备好了模型训练所有需要的要素，可以实现主要的训练过程部分了。理解这段代码⾄关重要，因为从事深度学习后，你会⼀遍⼜⼀遍地看到⼏乎相同的训练过程。在每次迭代中，我们读取⼀⼩批量训练样本，并通过我们的模型来获得⼀组预测。计算完损失后，我们开始反向传播，存储每个参数的梯度。最后，我们调⽤优化算法sgd来更新模型参数。

   

   概括⼀下，我们将执⾏以下循环：

   •  初始化参数

   •  重复以下训练，直到完成

   * 计算梯度

   * 更新参数

   在每个迭代周期（epoch）中，我们使⽤data_iter函数遍历整个数据集，并将训练数据集中所有样本都使⽤⼀次（假设样本数能够被批量⼤⼩整除）。这⾥的迭代周期个数num_epochs和学习率lr都是超参数，分别设为3和0.03。设置超参数很棘⼿，需要通过反复试验进⾏调整。

   ```python3
   for epoch in range(num_epochs):
   for X, y in data_iter(batch_size, features, labels):
   l = loss(net(X, w, b), y) # X和y的⼩批量损失
   # 因为l形状是(batch_size,1)，⽽不是⼀个标量。l中的所有元素被加到⼀起，
   # 并以此计算关于[w,b]的梯度
   l.sum().backward()
   sgd([w, b], lr, batch_size) # 使⽤参数的梯度更新参数
   with torch.no_grad():
   train_l = loss(net(features, w, b), labels)
   print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')
   ```

   