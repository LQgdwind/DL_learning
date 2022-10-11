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


### Chapter 3.3 线性回归的简洁实现

### Chapter 3.4 softmax回归

### Chapter 3.5 图像分类数据集

### Chapter 3.6 softmax回归的从零开始实现

### Chapter 3.7 softmax回归的简洁实现



## Chapter 4 MLP

### Chapter 4.1 多层感知机 

最简单的深度⽹络称为多层感知机。多层感知机由多层神经元组成，每⼀层与它的上⼀层相连，从中接收输⼊；同时每⼀层也与它的下⼀层相连，影响当前层的神经元。

我们可以通过在⽹络中加⼊⼀个或多个隐藏层来克服线性模型的限制，使其能处理更普遍的函数关系类型。要做到这⼀点，最简单的⽅法是将许多全连接层堆叠在⼀起。每⼀层都输出到上⾯的层，直到⽣成最后的输出。我们可以把前L-1层看作表示，把最后⼀层看作线性预测器。这种架构通常称为多层感知机（multilayer perceptron），通常缩写为*MLP*。

如果不加激活函数，我们会发现实际上多层的隐藏层堆叠等价于一个放射变换，这增加了参数开销且没有任何意义。所以我们引入了非线性的激活函数。一般来说，引入了非线性的激活函数之后多层感知机将不会退化为线性模型。

<b>常见的激活函数</b>

1. Relu激活函数

   ReLU(*x*) = max(*x,* 0)

   ReLU函数通过将相应的活性值设为0，仅保留正元素并丢弃所有负元素。

   使⽤ReLU的原因是，它求导表现得特别好：要么让参数消失，要么让参数通过。这使得优化表现得更好，并且ReLU减轻了困扰以往神经⽹络的梯度消失问题。

2. Sigmoid激活函数

   对于⼀个定义域在R中的输⼊，*sigmoid*函数将输⼊变换为区间(0, 1)上的输出。因此，sigmoid通常称为挤压函数（squashing function）：它将范围（-inf, inf）中的任意输⼊压缩到区间（0, 1）中的某个值：

3. tanh激活函数

   与sigmoid函数类似，tanh(双曲正切)函数也能将其输⼊压缩转换到区间(-1, 1)上。

### Chapter 4.2 多层感知机的从零开始实现

### Chapter 4.3

### Chapter 4.4

### Chapter 4.5 权重衰减(正则化技术)

**正则化是用来防止模型过拟合而采取的手段**。

我们对代价函数增加一个限制条件，**限制其较高次的参数大小不能过大**。还是使用回归模型举例子：

![\large h_{\theta}(x)=\theta_{0}+\theta_{1} x_{1}+\theta_{2} x_{2}^{2}+\theta_{3} x_{3}^{3}+\theta_{4} x_{4}^{4}](https://private.codecogs.com/gif.latex?%5Cinline%20%5Cbg_black%20%5Clarge%20h_%7B%5Ctheta%7D%28x%29%3D%5Ctheta_%7B0%7D&plus;%5Ctheta_%7B1%7D%20x_%7B1%7D&plus;%5Ctheta_%7B2%7D%20x_%7B2%7D%5E%7B2%7D&plus;%5Ctheta_%7B3%7D%20x_%7B3%7D%5E%7B3%7D&plus;%5Ctheta_%7B4%7D%20x_%7B4%7D%5E%7B4%7D)

正是那些高次项导致了过拟合的产生，所以如果我们能让这些高次项的系数接近于0的话，我们就能很好的拟合了。



**L2 正则化**

权重衰减（weight decay）是最⼴泛使⽤的正则化的技术之⼀，它通常也被称为*L*2正则化。

所谓权重衰减，就是指在损失函数中增加一个0.5 * λ * w^2。这样在更新权重的时候，我们就会额外减去一个λ * w。

L2正则化对于绝对值较大的权重予以很重的惩罚，对于绝对值很小的权重予以非常非常小的惩罚，当权重绝对值趋近于0时，基本不惩罚。这个性质与L2的平方项有关系，即越大的数，其平方越大，越小的数，比如小于1的数，其平方反而越小。

同时，他有另一个优势，在使用正规方程时，解析式中的逆始终存在的。

我们仅考虑惩罚项，优化算法在训练的每⼀步衰减权重。与特征选择相⽐，权重衰减为我们提供了⼀种连续的机制来调整函数的复杂度。较⼩的*λ*值对应较少约束的**w**，⽽较⼤的*λ*值对**w**的约束更⼤。

是否对相应的偏置*b*2进⾏惩罚在不同的实践中会有所不同，在神经⽹络的不同层中也会有所不同。通常，⽹络输出层的偏置项(也就是b)不会被正则化。

**L1正则化**
随着海量数据处理的兴起，工程上对于模型稀疏化的要求也随之出现了。这时候，L2正则化已经不能满足需求，因为它只是使得模型的参数值趋近于0，而不是等于0，这样就无法丢掉模型里的任何一个特征，因此无法做到稀疏化。这时，L1的作用随之显现。L1正则化的作用是使得大部分模型参数的值等于0，这样一来，当模型训练好后，这些权值等于0的特征可以省去，从而达到稀疏化的目的，也节省了存储的空间，因为在计算时，值为0的特征都可以不用存储了。

L1正则化对于所有权重予以同样的惩罚，也就是说，不管模型参数的大小，对它们都施加同等力度的惩罚，因此，较小的权重在被惩罚后，就会变成0。因此，在经过L1正则化后，大量模型参数的值变为0或趋近于0，当然也有一部分参数的值飙得很高。由于大量模型参数变为0，这些参数就不会出现在最终的模型中，因此达到了稀疏化的作用，这也说明了L1正则化自带特征选择的功能，这一点十分有用。



**L1正则化和L2正则化在实际应用中的比较**
L1在确实需要稀疏化模型的场景下，才能发挥很好的作用并且效果远胜于L2。在模型特征个数远大于训练样本数的情况下，如果我们事先知道模型的特征中只有少量相关特征（即参数值不为0），并且相关特征的个数少于训练样本数，那么L1的效果远好于L2。然而，需要注意的是，当相关特征数远大于训练样本数时，无论是L1还是L2，都无法取得很好的效果。

### Chapter 4.6 暂退法(dropout)

暂退法在前向传播过程中，计算每⼀内部层的同时注⼊噪声，这已经成为训练神经⽹络的常⽤技术。这种⽅法之所以被称为暂退法，因为我们从表⾯上看是在训练过程中丢弃（drop out）⼀些神经元。在整个训练过程的每⼀次迭代中，标准暂退法包括在计算下⼀层之前将当前层中的⼀些节点置零。

关键的挑战就是如何注⼊这种噪声。⼀种想法是以⼀种⽆偏向（unbiased）的⽅式注⼊噪声。这样在固定住其他层时，每⼀层的期望值等于没有噪⾳时的值。

在毕晓普的⼯作中，他将⾼斯噪声添加到线性模型的输⼊中。在每次训练迭代中，他将从均值为零的分布*ϵ* *∼* *N* (0*, σ*<sup>2</sup>) 采样噪声添加到输⼊**x**，从⽽产⽣扰动点**x**′ = **x** + *ϵ*，预期是*E*[**x**′] = **x**。

在标准暂退法正则化中，通过按保留（未丢弃）的节点的分数进⾏规范化来消除每⼀层的偏差。中间活性值*h*以暂退概率*p*由随机变量h'替换，如下所示:

**h'=0,  概率为P**

**h'= h/(1−p) ,  概率为1-p**

其期望值保持不变，即***E*[h‘] = h**。



通常，我们在**测试时不⽤暂退法**。给定⼀个训练好的模型和⼀个新的样本，我们不会丢弃任何节点，因此不需要标准化。



### Chapter 4.7 前向传播、反向传播和计算图

### Chapter 4.8 数值稳定性和模型初始化

初始化⽅案的选择在神经⽹络学习中起着举⾜轻重的作⽤，它对保持数值稳定性⾄关重要。此外，这些初始化⽅案的选择可以与⾮线性激活函数的选择有趣的结合在⼀起。我们选择哪个函数以及如何初始化参数可以决定优化算法收敛的速度有多快。糟糕选择可能会导致我们在训练时遇到梯度爆炸或梯度消失。

解决（或⾄少减轻）上述问题的⼀种⽅法是进⾏参数初始化，优化期间的注意和适当的正则化也可以进⼀步提⾼稳定性。

**默认初始化**

在前⾯的部分中，例如在 3.3节中，我们使⽤正态分布来初始化权重值。如果我们不指定初始化⽅法，框架将使⽤默认的随机初始化⽅法，对于中等难度的问题，这种⽅法通常很有效。

**Xavier初始化**

通常，Xavier初始化从均值为零，⽅差 *σ*<sup>2</sup> = 2/(*n*<sub>in</sub>+*n*<sub>out</sub>)的⾼斯分布中采样权重。

### Chapter 4.9  环境和分布偏移

### Chapter 4.10 Kaggle入门

## Chapter 5 深度学习计算

### Chapter 5.1 层和块

我们简要总结⼀下每个块必须提供的基本功能：

1. 将输⼊数据作为其前向传播函数的参数。

2. 通过前向传播函数来⽣成输出。请注意，输出的形状可能与输⼊的形状不同。

3. 计算其输出关于输⼊的梯度，可通过其反向传播函数进⾏访问。通常这是⾃动发⽣的。

4. 存储和访问前向传播计算所需的参数。

5. 根据需要初始化模型参数。

```python3
class MLP(nn.Module):
# ⽤模型参数声明层。这⾥，我们声明两个全连接的层
	def __init__(self):
# 调⽤MLP的⽗类Module的构造函数来执⾏必要的初始化。
# 这样，在类实例化时也可以指定其他函数参数，例如模型参数params（稍后将介绍）
		super().__init__()
		self.hidden = nn.Linear(20, 256) # 隐藏层
		self.out = nn.Linear(256, 10) # 输出层
# 定义模型的前向传播，即如何根据输⼊X返回所需的模型输出
	def forward(self, X):
# 注意，这⾥我们使⽤ReLU的函数版本，其在nn.functional模块中定义。
		return self.out(F.relu(self.hidden(X)))
```



