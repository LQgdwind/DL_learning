

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

### Chapter 4.3 多层感知机的简洁实现

### Chapter 4.4 模型选择，过拟合与欠拟合

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



现在我们可以更仔细地看看Sequential类是如何⼯作的，回想⼀下Sequential的设计是为了把其他模块串起来。为了构建我们⾃⼰的简化的MySequential，我们只需要定义两个关键函数：

1. ⼀种将块逐个追加到列表中的函数。

2. ⼀种前向传播函数，⽤于将输⼊按追加块的顺序传递给块组成的“链条”。下⾯的MySequential类提供了与默认Sequential类相同的功能。

```python3
class MySequential(nn.Module):
	def __init__(self, *args):
		super().__init__()
		for idx, module in enumerate(args):
			# 这⾥，module是Module⼦类的⼀个实例。我们把它保存在'Module'类的成员
			# 变量_modules中。module的类型是OrderedDict
			self._modules[str(idx)] = module
	def forward(self, X):
		# OrderedDict保证了按照成员添加的顺序遍历它们
		for block in self._modules.values():
			X = block(X)
		return X
```





到⽬前为⽌，我们⽹络中的所有操作都对⽹络的激活值及⽹络的参数起作⽤。然⽽，有时我们可能希望合并既不是上⼀层的结果也不是可更新参数的项，我们称之为常数参数（constant parameter）。

例如，我们需要⼀个计算函数 *f*(**x**, **w**) = *c* *·* **w**<sup>⊤</sup>**x**的层，其中**x**是输⼊，**w**是参数，c是某个在优化过程中没有更新的指定常量。

因此我们实现了⼀个FixedHiddenMLP类，如下所⽰：

```python3
class FixedHiddenMLP(nn.Module):
	def __init__(self):
		super().__init__()
		# 不计算梯度的随机权重参数。因此其在训练期间保持不变
		self.rand_weight = torch.rand((20, 20), requires_grad=False)
		self.linear = nn.Linear(20, 20)
	def forward(self, X):
		X = self.linear(X)
		# 使⽤创建的常量参数以及relu和mm函数
		X = F.relu(torch.mm(X, self.rand_weight) + 1) # 复⽤全连接层。这相当于两个全连接层共享参数
		X = self.linear(X)
		# 控制流
		while X.abs().sum() > 1: #L1范数
            X /= 2
		return X.sum()
```



### Chapter 5.2 参数管理

**参数访问**

我们从已有模型中访问参数。当通过Sequential类定义模型时，我们可以通过索引来访问模型的任意层。这就像模型是⼀个列表⼀样，每层的参数都在其属性中。如下所⽰，我们可以检查第⼆个全连接层的参数。

```python3
print(net[2].state_dict())

# OrderedDict([('weight', tensor([[ 0.0743, 0.1876, 0.0571, 0.3447, 0.3483, -0.2867, 0.3273, -0.1527]])), ('bias', tensor([0.1162]))])
```

下⾯的代码从第⼆个全连接层（即第三个神经⽹络层）提取偏置，提取后返回的是⼀个参数类实例，并进⼀步访问该参数的值。

```python3
print(type(net[2].bias))
print(net[2].bias)
print(net[2].bias.data)

# <class 'torch.nn.parameter.Parameter'>
# Parameter containing:
# tensor([0.1162], requires_grad=True)
# tensor([0.1162])
```

同样的我们可以访问梯度

```python3
net[2].weight.grad == None
```



**参数初始化**

内置初始化

```python3
net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
# 偏置矩阵存的是转置之前的矩阵，与Linear形状刚好相反 比如net[0].weight的形状就是(8,4) 
# 原因很显然 要把输入的特征数4变成输出的特征数8，必须进行8次线性变换得到8个特征。
def init_normal(m):
	if type(m) == nn.Linear:
		nn.init.normal_(m.weight, mean=0, std=0.01)
		nn.init.zeros_(m.bias)
net.apply(init_normal)
print(net[0].weight.data[0], net[0].bias.data[0])

def init_constant(m):
	if type(m) == nn.Linear:
		nn.init.constant_(m.weight, 1)
		nn.init.zeros_(m.bias)
net.apply(init_constant)
print(net[0].weight.data[0], net[0].bias.data[0])

# 我们可以对某些块运用不同的初始化办法
def xavier(m):
	if type(m) == nn.Linear:
		nn.init.xavier_uniform_(m.weight)
def init_42(m):
	if type(m) == nn.Linear:
		nn.init.constant_(m.weight, 42)

net[0].apply(xavier)
net[2].apply(init_42)
print(net[0].weight.data[0])
print(net[2].weight.data)
# tensor([-0.3261, -0.5587, 0.0063, -0.3914])
# tensor([[42., 42., 42., 42., 42., 42., 42., 42.]])
```



参数绑定

```python3
# 我们需要给共享层⼀个名称，以便可以引⽤它的参数
shared = nn.Linear(8, 8)
net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(),
shared, nn.ReLU(),
shared, nn.ReLU(),
nn.Linear(8, 1))
net(X)
# 检查参数是否相同
print(net[2].weight.data[0] == net[4].weight.data[0])
net[2].weight.data[0, 0] = 100
# 确保它们实际上是同⼀个对象，⽽不只是有相同的值
print(net[2].weight.data[0] == net[4].weight.data[0])

# tensor([True, True, True, True, True, True, True, True])
# tensor([True, True, True, True, True, True, True, True])
```

这个例⼦表明第三个和第五个神经⽹络层的参数是绑定的。它们不仅值相等，⽽且由相同的张量表⽰。

因此，如果我们改变其中⼀个参数，另⼀个参数也会改变。你可能会思考：当参数绑定时，梯度会发⽣什么情况？

答案是由于模型参数包含梯度，因此在反向传播期间第⼆个隐藏层（即第三个神经⽹络层）和第三个隐藏层（即第五个神经⽹络层）的梯度会加在⼀起。

### Chapter 5.3 延后初始化

### Chapter 5.4 自定义层

```python3
class MyLinear(nn.Module):
	def __init__(self, in_units, units):
		super().__init__()
		self.weight = nn.Parameter(torch.randn(in_units, units))
		self.bias = nn.Parameter(torch.randn(units,))
	def forward(self, X):
		linear = torch.matmul(X, self.weight.data) + self.bias.data
		return F.relu(linear)
```

我们还可以使⽤⾃定义层构建模型，就像使⽤内置的全连接层⼀样使⽤⾃定义层。

```python3
net = nn.Sequential(MyLinear(64, 8), MyLinear(8, 1))
net(torch.rand(2, 64))
```

### Chapter 5.5 读写文件

对于单个张量，我们可以直接调⽤load和save函数分别读写它们。这两个函数都要求我们提供⼀个名称，save要求将要保存的变量作为输⼊。

```python3
import torch
from torch import nn
from torch.nn import functional as F x = torch.arange(4)
torch.save(x, 'x-file')
x2 = torch.load('x-file')
print(x2)
```

保存单个权重向量（或其他张量）确实有⽤，但是如果我们想保存整个模型，并在以后加载它们，单独保存每个向量则会变得很⿇烦。毕竟，我们可能有数百个参数散布在各处。因此，深度学习框架提供了内置函数来保存和加载整个⽹络。需要注意的⼀个重要细节是，这将保存模型的参数⽽不是保存整个模型。

```python3
class MLP(nn.Module):
def __init__(self):
super().__init__()
self.hidden = nn.Linear(20, 256)
self.output = nn.Linear(256, 10)
def forward(self, x):
return self.output(F.relu(self.hidden(x)))
net = MLP()
X = torch.randn(size=(2, 20))
Y = net(X)

torch.save(net.state_dict(), 'mlp.params')

clone = MLP()
clone.load_state_dict(torch.load('mlp.params'))
print(clone.eval())
```

由于两个实例具有相同的模型参数，在输⼊相同的X时，两个实例的计算结果应该相同。



### Chapter 5.6 GPU



## Chapter 6 卷积神经网络

卷积神经⽹络（convolutional neural network，CNN）是⼀类强⼤的、为处理图像数据⽽设计的神经⽹络。CNN利⽤先验知识，即利⽤相近像素之间的相互关联性，从图像数据中学习得到有效的模型。基于卷积神经⽹络架构的模型在计算机视觉领域中已经占主导地位，当今⼏乎所有的图像识别、⽬标检测或语义分割相关的学术竞赛和商业应⽤都以这种⽅法为基础。

现代卷积神经⽹络的设计得益于⽣物学、群论和⼀系列的补充实验。卷积神经⽹络需要的参数少于全连接架构的⽹络，⽽且卷积也很容易⽤GPU并⾏计算。因此卷积神经⽹络除了能够⾼效地采样从⽽获得精确的模型，还能够⾼效地计算。



### Chapter 6.1 从全连接层到卷积

相比于FC(fully connected layer)，卷积神经⽹络正是将空间不变性（spatial invariance）的这⼀概念系统化，从⽽基于这个模型使⽤较少的参数来学习有⽤的表⽰。

1. 平移不变性（translation invariance）：不管检测对象出现在图像中的哪个位置，神经⽹络的前⾯⼏层应该对相同的图像区域具有相似的反应，即为“平移不变性”。

2. 局部性（locality）：神经⽹络的前⾯⼏层应该只探索输⼊图像中的局部区域，⽽不过度在意图像中相隔较远区域的关系，这就是“局部性”原则。最终，可以聚合这些局部特征，以在整个图像级别进⾏预测。



在深度学习研究社区中，**V**被称为卷积核（convolution kernel）或者滤波器（filter），亦或简单地称之为该卷积层的权重，通常该权重是可学习的参数。

当图像处理的局部区域很⼩时，卷积神经⽹络与多层感知机的训练差异可能是巨⼤的：以前，多层感知机可能需要数⼗亿个参数来表⽰⽹络中的⼀层，⽽现在卷积神经⽹络通常只需要⼏百个参数，⽽且不需要改变输⼊或隐藏表⽰的维数。参数⼤幅减少的代价是，我们的特征现在是平移不变的，并且当确定每个隐藏活性值时，每⼀层只包含局部的信息。

以上所有的权重学习都将依赖于归纳偏置。当这种偏置与现实相符时，我们就能得到样本有效的模型，并且这些模型能很好地泛化到未知数据中。

但如果这偏置与现实不符时，⽐如当图像不满⾜平移不变时，我们的模型可能难以拟合我们的训练数据。



然⽽这种⽅法有⼀个问题：我们忽略了图像⼀般包含三个通道/三种原⾊（红⾊、绿⾊和蓝⾊）。

实际上，图像不是⼆维张量，⽽是⼀个由⾼度、宽度和颜⾊组成的三维张量，⽐如包含1024 *×* 1024 *×* 3个像素。

前两个轴与像素的空间位置有关，⽽第三个轴可以看作是每个像素的多维表⽰。因此，我们将X索引为[X]<sub>*i,j,k*</sub>。由此卷积相应地调整为[V]<sub>*a,b,c*</sub>，⽽不是[**V**]<sub>*a,b*</sub>。此外，由于输⼊图像是三维的，我们的隐藏表⽰H也最好采⽤三维张量。换句话说，对于每⼀个空间位置，我们想要采⽤⼀组⽽不是⼀个隐藏表⽰。这样⼀组隐藏表⽰可以想象成⼀些互相堆叠的⼆维⽹格。因此，我们可以把隐藏表⽰想象为⼀系列具有⼆维张量的通道（channel）。这些通道有时也被称为特征映射（feature maps），因为每个通道都向后续层提供⼀组空间化的学习特征。直观上你可以想象在靠近输⼊的底层，⼀些通道专⻔识别边缘，⽽⼀些通道专⻔识别纹理。为了⽀持输⼊X和隐藏表⽰H中的多个通道，我们可以在V中添加第四个坐标，即[V]<sub>*a,b,c,d*</sub>。



综上所述，卷积层需要学习的参数比全连接层少得多，**对于平面图像，卷积核通常是四维的，分别是输入通道a，输出通道b，卷积核size:c*d。**

一般来说，输出通道数等于卷积核的个数

不考虑bias，需要训练的权重参数量为 ***a*b*c*d**



### Chapter 6.2 图像卷积

严格来说，卷积层是个错误的叫法，因为它所表达的运算其实是互相关运算（cross-correlation），⽽不是卷积运算。

在⼆维互相关运算中，卷积窗⼝从输⼊张量的左上⻆开始，从左到右、从上到下滑动。当卷积窗⼝滑动到新⼀个位置时，包含在该窗⼝中的部分张量与卷积核张量进⾏按元素相乘，得到的张量再求和得到⼀个单⼀的标量值，由此我们得出了这⼀位置的输出张量值。

在单通道输入单通道输出的情况下，输出大小等于输入大小*n*<sub>h</sub> *×* *n*<sub>w</sub>减去卷积核大小*k*<sub>h</sub>*×* *k*<sub>w</sub>：

**输出大小  =  (*n*<sub>*h*</sub> *−* *k*<sub>*h*</sub> + 1) *×* (*n*<sub>*w*</sub> *−* *k*<sub>*w*</sub> + 1)**



**特征映射**

输出的卷积层有时被称为特征映射（feature map），因为它可以被视为⼀个输⼊映射到下⼀层的空间维度的转换器。

**感受野**

在卷积神经⽹络中，对于某⼀层的任意元素*x*，其感受野（receptivefield）是指在前向传播期间可能影响*x*计算的所有元素（来⾃所有先前层）。

请注意，感受野可能⼤于输⼊的实际⼤⼩。

举一个例子来解释感受野：给定2 *×* 2卷积核，阴影输出元素值19的感受野是输⼊阴影部分的四个元素。假设之前输出为**Y**，其⼤⼩为2 *×* 2，现在我们在其后附加⼀个卷积层，该卷积层以**Y**为输⼊，输出单个元素*z*。在这种情况下，**Y**上的*z*的感受野包括**Y**的所有四个元素，⽽输⼊的感受野包括最初所有九个输⼊元素。

因此，当⼀个特征图中的任意元素需要检测更⼴区域的输⼊特征时，我们可以构建⼀个更深的⽹络。



**卷积层与学习卷积核**

卷积层对输⼊和卷积核权重进⾏互相关运算，并在添加标量偏置之后产⽣输出。所以，卷积层中的两个被训练的参数是卷积核权重和标量偏置。就像我们之前随机初始化全连接层⼀样，在训练基于卷积层的模型时，我们也随机初始化卷积核权重。

```python3
def corr2d(X, K): #@save
	"""计算⼆维互相关运算"""
    h, w = K.shape
	Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
	for i in range(Y.shape[0]):
		for j in range(Y.shape[1]):
			Y[i, j] = (X[i:i + h, j:j + w] * K).sum()
	return Y

class Conv2D(nn.Module):
	def __init__(self, kernel_size):
		super().__init__()
		self.weight = nn.Parameter(torch.rand(kernel_size))
		self.bias = nn.Parameter(torch.zeros(1))
	def forward(self, x):
		return corr2d(x, self.weight) + self.bias
```

以下为卷积核参数的学习过程:

```python3
# 构造⼀个⼆维卷积层，它具有1个输出通道和形状为（1，2）的卷积核
conv2d = nn.Conv2d(1,1, kernel_size=(1, 2), bias=False) 
# 这个⼆维卷积层使⽤四维输⼊和输出格式（批量⼤⼩、通道、⾼度、宽度），
# 其中批量⼤⼩和通道数都为1 X = X.reshape((1, 1, 6, 8))
Y = Y.reshape((1, 1, 6, 7))
lr = 3e-2 # 学习率

for i in range(10):
	Y_hat = conv2d(X)
	l = (Y_hat - Y) ** 2
	conv2d.zero_grad()
	l.sum().backward()
	# 迭代卷积核
	conv2d.weight.data[:] -= lr * conv2d.weight.grad
	if (i + 1) % 2 == 0:
		print(f'epoch {i+1}, loss {l.sum():.3f}')
```

以下为学习输出

```python3
epoch 2, loss 11.205
epoch 4, loss 3.162
epoch 6, loss 1.056
epoch 8, loss 0.392
epoch 10, loss 0.154
```

### Chapter 6.3 填充(padding)和步幅(stride)

我们将每次滑动元素的数量称为步幅（stride）,在输⼊图像的边界填充元素（通常填充元素是0）称为填充(padding)。

在许多情况下，我们需要设置*p*<sub>*h*</sub> = *k*<sub>*h*</sub> *−* 1和*p*<sub>*w*</sub> = *k*<sub>*w*</sub> *−* 1，使输⼊和输出具有相同的⾼度和宽度。这样可以在构建⽹络时更容易地预测每个图层的输出形状。假设*k*<sub>*h*</sub>是奇数，我们将在⾼度的两侧填充*p*<sub>*h*</sub>/2⾏。如果*k*<sub>*h*</sub>是偶数，则⼀种可能性是在输⼊顶部填充⌈*p*<sub>*h*</sub>/2⌉⾏，在底部填充⌊*p*<sub>*h*</sub>/2⌋⾏。同理，我们填充宽度的两侧。卷积神经⽹络中卷积核的⾼度和宽度通常为奇数，例如1、3、5或7。选择奇数的好处是，保持空间维度的同时，我们可以在顶部和底部填充相同数量的⾏，在左侧和右侧填充相同数量的列。

此外，使⽤奇数的核大小和填充大小也提供了书写上的便利。对于任何⼆维张量X，当满⾜：1. 卷积核的大小是奇数；2. 所有边的填充⾏数和列数相同；3. 输出与输⼊具有相同⾼度和宽度则可以得出：输出Y<sub>[i, j]</sub>是通过以输⼊X<sub>[i, j]</sub>为中⼼，与卷积核进⾏互相关计算得到的。



通常，当垂直填充为*p*<sub>*h*</sub> 、垂直步幅为*s*<sub>*h*</sub>、水平填充为*p*<sub>*w*</sub>、⽔平步幅为*s*<sub>*w*</sub>时

输出形状:<font color = "red">  **⌊(*n*<sub>*h*</sub> *−* *k*<sub>*h*</sub> + *p*<sub>*h*</sub> + *s*<sub>*h*</sub>)/*s*<sub>*h*</sub>⌋ × ⌊(*n*<sub>*w*</sub> *−* *k*<sub>*w*</sub> + *p*<sub>*w*</sub> + *s*<sub>*w*</sub>)/*s*<sub>*w*</sub>⌋*.***</font>



### Chapter 6.4 多输入多输出通道

**多输入通道**

当输⼊包含多个通道时，需要构造⼀个与输⼊数据具有相同输⼊通道数的卷积核，以便与输⼊数据进⾏互相关运算。假设输⼊的通道数为*c*<sub>*i*</sub>，那么卷积核的输⼊通道数也需要为*c*<sub>*i*</sub>。如果卷积核的窗⼝形状是*k*<sub>*h*</sub> *×* *k*<sub>*w*</sub>，那么当*c*<sub>*i*</sub> = 1时，我们可以把卷积核看作形状为*k*<sub>*h*</sub> *×* *k*<sub>*w*</sub>的⼆维张量。

然⽽，当*c*<sub>*i*</sub> *>* 1时，我们卷积核的每个输⼊通道将包含形状为*k*<sub>*h*</sub> *×* *k*<sub>*w*</sub>的张量。将这些张量*c*<sub>*i*</sub>连结在⼀起可以得到形状为*c*<sub>*i* </sub>*×* *k*<sub>*h* </sub>*×* *k*<sub>*w*</sub>的卷积核。由于输⼊和卷积核都有*c*<sub>*i*</sub>个通道，我们可以对每个通道输⼊的⼆维张量和卷积核的⼆维张量进⾏互相关运算，再对通道求和（将*c*<sub>*i*</sub>的结果相加）得到⼆维张量。这是多通道输⼊和多输⼊通道卷积核之间进⾏⼆维互相关运算的结果。

**多输出通道**

在最流⾏的神经⽹络架构中，随着神经⽹络层数的加深，我们常会增加输出通道的维数，通过减少空间分辨率以获得更⼤的通道深度。直观地说，我们可以将每个通道看作是对不同特征的响应。⽽现实可能更为复杂⼀些，因为每个通道不是独⽴学习的，⽽是为了共同使⽤⽽优化的。因此，多输出通道并不仅是学习多个单通道的检测器。

⽤*c*<sub>*i*</sub>和*c*<sub>*o*</sub>分别表⽰输⼊和输出通道的数⽬，并让*k*<sub>*h*</sub>和*k*<sub>*w*</sub>为卷积核的⾼度和宽度。为了获得多个通道的输出，我们可以为每个输出通道创建⼀个形状为*c*<sub>*i* </sub>*×* *k*<sub>*h* </sub>*×* *k*<sub>*w*</sub>的卷积核张量，这样卷积核的形状是*c*<sub>*o* </sub>*×* *c*<sub>*i* </sub>*×* *k*<sub>*h* </sub>*×* *k*<sub>*w*</sub>。在互相关运算中，每个输出通道先获取所有输⼊通道，再以对应该输出通道的卷积核计算出结果。



<font color = "red">总的来说，每多一个输入通道卷积层就会多一层，每多一个输出通道就会多一个卷积核。</font>



**1×1卷积核**

我们可以通过改变1×1卷积核的数目来改变卷积层的通道数量。卷积核每多一层输出通道数就会多一个。



### Chapter 6.5 汇聚层(池化层)

通常当我们处理图像时，我们希望逐渐降低隐藏表⽰的空间分辨率、聚集信息，这样随着我们在神经⽹络中层叠的上升，每个神经元对其敏感的感受野（输⼊）就越⼤。

⽽我们的机器学习任务通常会跟全局图像的问题有关（例如，“图像是否包含⼀只猫呢？”），所以我们最后⼀层的神经元应该对整个输⼊的全局敏感。通过逐渐聚合信息，⽣成越来越粗糙的映射，最终实现学习全局表⽰的⽬标，同时将卷积图层的所有优势保留在中间层。

汇聚（pooling）层，它具有双重⽬的：

1. 降低卷积层对位置的敏感性。
2. 降低对空间降采样表⽰的敏感性。



与卷积层类似，汇聚层运算符由⼀个固定形状的窗⼝组成，该窗⼝根据其步幅⼤⼩在输⼊的所有区域上滑动，为固定形状窗⼝（有时称为汇聚窗⼝）遍历的每个位置计算⼀个输出。然⽽，不同于卷积层中的输⼊与卷积核之间的互相关计算，汇聚层不包含参数。相反，池运算是确定性的，我们通常计算汇聚窗⼝中所有元素的最⼤值或平均值。这些操作分别称为最⼤汇聚层（maximum pooling）和平均汇聚层（average pooling）。

在这两种情况下，与互相关运算符⼀样，汇聚窗⼝从输⼊张量的左上⻆开始，从左往右、从上往下的在输⼊张量内滑动。在汇聚窗⼝到达的每个位置，它计算该窗⼝中输⼊⼦张量的最⼤值或平均值。计算最⼤值或平均值是取决于使⽤了最⼤汇聚层还是平均汇聚层。



总结：

1. 对于给定输⼊元素，最⼤汇聚层会输出该窗⼝内的最⼤值，平均汇聚层会输出该窗⼝内的平均值。

2. 汇聚层的主要优点之⼀是减轻卷积层对位置的过度敏感。

3. 我们可以指定汇聚层的填充和步幅。

4. 使⽤最⼤汇聚层以及⼤于1的步幅，可减少空间维度（如⾼度和宽度）。

5. 汇聚层的输出通道数与输⼊通道数相同。



### Chapter 6.6 Lenet简介

• LeNet是最早发布的卷积神经⽹络之⼀。

• 卷积神经⽹络（CNN）是⼀类使⽤卷积层的⽹络。

• 在卷积神经⽹络中，我们组合使⽤卷积层、⾮线性激活函数和汇聚层。

• 为了构造⾼性能的卷积神经⽹络，我们通常对卷积层进⾏排列，逐渐降低其表⽰的空间分辨率，同时增

加通道数。

• 在传统的卷积神经⽹络中，卷积块编码得到的表征在输出之前需由⼀个或多个全连接层进⾏处理。



## Chapter 7 现代卷积神经网络

### Chapter 7.1 深度卷积神经⽹络（**AlexNet**）

