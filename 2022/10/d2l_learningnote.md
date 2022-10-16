

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

AlexNet由⼋层组成：五个卷积层、两个全连接隐藏层和⼀个全连接输出层。

```python3
import torch
from torch import nn
from d2l import torch as d2l
net = nn.Sequential(
	# 这⾥，我们使⽤⼀个11*11的更⼤窗⼝来捕捉对象。
	# 同时，步幅为4，以减少输出的⾼度和宽度。
	# 另外，输出通道的数⽬远⼤于LeNet
	nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=1), nn.ReLU(),
	nn.MaxPool2d(kernel_size=3, stride=2),
	# 减⼩卷积窗⼝，使⽤填充为2来使得输⼊与输出的⾼和宽⼀致，且增⼤输出通道数
	nn.Conv2d(96, 256, kernel_size=5, padding=2), nn.ReLU(),
	nn.MaxPool2d(kernel_size=3, stride=2),
	# 使⽤三个连续的卷积层和较⼩的卷积窗⼝。
	# 除了最后的卷积层，输出通道的数量进⼀步增加。
	# 在前两个卷积层之后，汇聚层不⽤于减少输⼊的⾼度和宽度
	nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(),
	nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(),
	nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(),
	nn.MaxPool2d(kernel_size=3, stride=2),
	nn.Flatten(),
	# 这⾥，全连接层的输出数量是LeNet中的好⼏倍。使⽤dropout层来减轻过拟合
	nn.Linear(6400, 4096), nn.ReLU(),
	nn.Dropout(p=0.5),
	nn.Linear(4096, 4096), nn.ReLU(),
	nn.Dropout(p=0.5),
	# 最后是输出层。由于这⾥使⽤Fashion-MNIST，所以⽤类别数为10，⽽⾮论⽂中的1000
	nn.Linear(4096, 10))
```



### Chapter 7.2 使用块的网络(VGG)

经典卷积神经⽹络的基本组成部分是下⾯的这个序列：

1. 带填充以保持分辨率的卷积层；

2. ⾮线性激活函数，如ReLU；

3. 汇聚层，如最⼤汇聚层。

⽽⼀个VGG块与之类似，由⼀系列卷积层组成，后⾯再加上⽤于空间下采样的最⼤汇聚层。在最初的VGG论⽂中，作者使⽤了带有3 *×* 3卷积核、填充为1（保持⾼度和宽度）的卷积层，=和带有2 *×* 2汇聚窗⼝、步幅为2（每个块后的分辨率减半）的最⼤汇聚层。在下⾯的代码中，我们定义了⼀个名为vgg_block的函数来实现⼀个VGG块。

该函数有三个参数，分别对应于卷积层的数量num_convs、输⼊通道的数量in_channels 和输出通道的数量out_channels.

```python3
import torch
from torch import nn
from d2l import torch as d2l
def vgg_block(num_convs, in_channels, out_channels):
	layers = []
	for _ in range(num_convs):
		layers.append(nn.Conv2d(in_channels, out_channels,kernel_size=3, padding=1))
		layers.append(nn.ReLU())
		in_channels = out_channels
	layers.append(nn.MaxPool2d(kernel_size=2,stride=2))
	return nn.Sequential(*layers)
```

与AlexNet、LeNet⼀样，VGG⽹络可以分为两部分：第⼀部分主要由卷积层和汇聚层组成，第⼆部分由全连接层组成。

原始VGG⽹络有5个卷积块，其中前两个块各有⼀个卷积层，后三个块各包含两个卷积层。第⼀个模块有64个输出通道，每个后续模块将输出通道数量翻倍，直到该数字达到512。由于该⽹络使⽤8个卷积层和3个全连接层，因此它通常被称为VGG-11。

```python3
conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))

def vgg(conv_arch):
	conv_blks = []
	in_channels = 1 # 卷积层部分
	for (num_convs, out_channels) in conv_arch:
		conv_blks.append(vgg_block(num_convs, in_channels, out_channels))
		in_channels = out_channels
	return nn.Sequential(
		*conv_blks, nn.Flatten(),
		# 全连接层部分
		nn.Linear(out_channels * 7 * 7, 4096), nn.ReLU(), nn.Dropout(0.5),
		nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5),
		nn.Linear(4096, 10))
    
net = vgg(conv_arch)
```

总结：

1. VGG-11使⽤可复⽤的卷积块构造⽹络。不同的VGG模型可通过每个块中卷积层数量和输出通道数量的差异来定义。

2. 块的使⽤导致⽹络定义的⾮常简洁。使⽤块可以有效地设计复杂的⽹络。

3. 在VGG论⽂中，Simonyan和Ziserman尝试了各种架构。特别是他们发现深层且窄的卷积（即3 *×* 3）⽐较浅层且宽的卷积更有效。



### Chapter 7.3 网络中的网络(NiN)

LeNet、AlexNet和VGG都有⼀个共同的设计模式：通过⼀系列的卷积层与汇聚层来提取空间结构特征；然后通过全连接层对特征的表征进⾏处理。AlexNet和VGG对LeNet的改进主要在于如何扩⼤和加深这两个模块。或者，可以想象在这个过程的早期使⽤全连接层。然⽽，如果使⽤了全连接层，可能会完全放弃表征的空间结构。⽹络中的⽹络（*NiN*）提供了⼀个⾮常简单的解决⽅案：在每个像素的通道上分别使⽤多层感知机。

卷积层的输⼊和输出由四维张量组成，张量的每个轴分别对应样本、通道、⾼度和宽度。另外，全连接层的输⼊和输出通常是分别对应于样本和特征的⼆维张量。NiN的想法是在每个像素位置（针对每个⾼度和宽度）应⽤⼀个全连接层。如果我们将权重连接到每个空间位置，我们可以将其视为1 *×* 1卷积层，或作为在每个像素位置上独⽴作⽤的全连接层。从另⼀个⻆度看，即将空间维度中的每个像素视为单个样本，将通道维度视为不同特征（feature）。

NiN块以⼀个普通卷积层开始，后⾯是两个1 *×* 1的卷积层。这两个1 *×* 1卷积层充当带有ReLU激活函数的逐像素全连接层。第⼀层的卷积窗⼝形状通常由⽤⼾设置。随后的卷积窗⼝形状固定为1 *×* 1。

```python3
import torch
from torch import nn
from d2l import torch as d2l
def nin_block(in_channels, out_channels, kernel_size, strides, padding):
	return nn.Sequential(
		nn.Conv2d(in_channels, out_channels, kernel_size, strides, padding),
		nn.ReLU(),
		nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ReLU(),
		nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ReLU())

net = nn.Sequential(
	nin_block(1, 96, kernel_size=11, strides=4, padding=0),
	nn.MaxPool2d(3, stride=2),
	nin_block(96, 256, kernel_size=5, strides=1, padding=2),
	nn.MaxPool2d(3, stride=2),
	nin_block(256, 384, kernel_size=3, strides=1, padding=1),
	nn.MaxPool2d(3, stride=2),
	nn.Dropout(0.5),
	# 标签类别数是10
	nin_block(384, 10, kernel_size=3, strides=1, padding=1),
	nn.AdaptiveAvgPool2d((1, 1)),
	# 将四维的输出转成⼆维的输出，其形状为(批量⼤⼩,10)
	nn.Flatten())
```



总结：

1. NiN使⽤由⼀个卷积层和多个1 *×* 1卷积层组成的块。该块可以在卷积神经⽹络中使⽤，以允许更多的每像素⾮线性。

2. NiN去除了容易造成过拟合的全连接层，将它们替换为全局平均汇聚层（即在所有位置上进⾏求和）。该汇聚层通道数量为所需的输出数量（例如，Fashion-MNIST的输出为10）。

3. 移除全连接层可减少过拟合，同时显著减少NiN的参数。

4. NiN的设计影响了许多后续卷积神经⽹络的设计。、



### Chapter 7.4 并行连结的网络(GoogLeNet)

在GoogLeNet中，基本的卷积块被称为*Inception*块（Inception block）。

Inception块由四条并⾏路径组成。前三条路径使⽤窗⼝⼤⼩为1 *×* 1、3 *×* 3和5 *×* 5的卷积层，从不同空间⼤⼩中提取信息。中间的两条路径在输⼊上执⾏1 *×* 1卷积，以减少通道数，从⽽降低模型的复杂性。第四条路径使⽤3 *×* 3最⼤汇聚层，然后使⽤1 *×* 1卷积层来改变通道数。这四条路径都使⽤合适的填充来使输⼊与输出的⾼和宽⼀致，最后我们将每条线路的输出在通道维度上连结，并构成Inception块的输出。在Inception块中，通常调整的超参数是每层输出通道数。

```python3
class Inception(nn.Module):
# c1--c4是每条路径的输出通道数
	def __init__(self, in_channels, c1, c2, c3, c4, **kwargs):
		super(Inception, self).__init__(**kwargs)
		# 线路1，单1x1卷积层
		self.p1_1 = nn.Conv2d(in_channels, c1, kernel_size=1) # 线路2，1x1卷积层后接3x3卷积层
		self.p2_1 = nn.Conv2d(in_channels, c2[0], kernel_size=1)
		self.p2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1) # 线路3，1x1卷积层后接5x5卷积层
		self.p3_1 = nn.Conv2d(in_channels, c3[0], kernel_size=1)
		self.p3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2) # 线路4，3x3最⼤汇聚层后接1x1卷积层
		self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
		self.p4_2 = nn.Conv2d(in_channels, c4, kernel_size=1)
	def forward(self, x):
		p1 = F.relu(self.p1_1(x))
		p2 = F.relu(self.p2_2(F.relu(self.p2_1(x))))
		p3 = F.relu(self.p3_2(F.relu(self.p3_1(x))))
		p4 = F.relu(self.p4_2(self.p4_1(x)))
		# 在通道维度上连结输出
		return torch.cat((p1, p2, p3, p4), dim=1)
    
```

为什么GoogLeNet这个⽹络如此有效呢？⾸先我们考虑⼀下滤波器（filter）的组合，它们可以⽤各种滤波器尺⼨探索图像，这意味着不同⼤⼩的滤波器可以有效地识别不同范围的图像细节。同时，我们可以为不同的滤波器分配不同数量的参数。

GoogLeNet⼀共使⽤9个Inception块和全局平均汇聚层的堆叠来⽣成其估计值。Inception块之间的最⼤汇聚层可降低维度。第⼀个模块类似于AlexNet和LeNet，Inception块的组合从VGG继承，全局平均汇聚层避免了在最后使⽤全连接层。

现在，我们逐⼀实现GoogLeNet的每个模块。

第⼀个模块使⽤64个通道、7 *×* 7卷积层。

```python3
b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
		nn.ReLU(),
		nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
```

第⼆个模块使⽤两个卷积层：第⼀个卷积层是64个通道、1 *×* 1卷积层；第⼆个卷积层使⽤将通道数量增加三倍的3 *×* 3卷积层。这对应于Inception块中的第⼆条路径。

```python3
b2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1),
		nn.ReLU(),
		nn.Conv2d(64, 192, kernel_size=3, padding=1),
		nn.ReLU(),
		nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
```

第三个模块串联两个完整的Inception块。第⼀个Inception块的输出通道数为64 + 128 + 32 + 32 = 256，四个路径之间的输出通道数量⽐为64 : 128 : 32 : 32 = 2 : 4 : 1 : 1。第⼆个和第三个路径⾸先将输⼊通道的数量分别减少到96/192 = 1/2和16/192 = 1/12，然后连接第⼆个卷积层。第⼆个Inception块的输出通道数增加到128 + 192 + 96 + 64 = 480，四个路径之间的输出通道数量⽐为128 : 192 : 96 : 64 = 4 : 6 : 3 : 2。第⼆条和第三条路径⾸先将输⼊通道的数量分别减少到128/256 = 1/2和32/256 = 1/8。

```python3
b3 = nn.Sequential(Inception(192, 64, (96, 128), (16, 32), 32),
		Inception(256, 128, (128, 192), (32, 96), 64),
		nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
```

第四模块更加复杂，它串联了5个Inception块，其输出通道数分别是192 + 208 + 48 + 64 = 512、160 + 224 +64 + 64 = 512、128 + 256 + 64 + 64 = 512、112 + 288 + 64 + 64 = 528和256 + 320 + 128 + 128 = 832。这些路径的通道数分配和第三模块中的类似，⾸先是含3×3卷积层的第⼆条路径输出最多通道，其次是仅含1×1卷积层的第⼀条路径，之后是含5×5卷积层的第三条路径和含3×3最⼤汇聚层的第四条路径。其中第⼆、第三条路径都会先按⽐例减⼩通道数。这些⽐例在各个Inception块中都略有不同。

```python3
b4 = nn.Sequential(Inception(480, 192, (96, 208), (16, 48), 64),
		Inception(512, 160, (112, 224), (24, 64), 64),
		Inception(512, 128, (128, 256), (24, 64), 64),
		Inception(512, 112, (144, 288), (32, 64), 64),
		Inception(528, 256, (160, 320), (32, 128), 128),
		nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
```

第五模块包含输出通道数为256 + 320 + 128 + 128 = 832和384 + 384 + 128 + 128 = 1024的两个Inception块。其中每条路径通道数的分配思路和第三、第四模块中的⼀致，只是在具体数值上有所不同。需要注意的是，第五模块的后⾯紧跟输出层，该模块同NiN⼀样使⽤全局平均汇聚层，将每个通道的⾼和宽变成1。最后我们将输出变成⼆维数组，再接上⼀个输出个数为标签类别数的全连接层。

```python3
b5 = nn.Sequential(Inception(832, 256, (160, 320), (32, 128), 128),
		Inception(832, 384, (192, 384), (48, 128), 128),
		nn.AdaptiveAvgPool2d((1,1)),
        # 全局平均汇聚层
		nn.Flatten())

net = nn.Sequential(b1, b2, b3, b4, b5, nn.Linear(1024, 10))
```



总结：

1.  Inception块相当于⼀个有4条路径的⼦⽹络。它通过不同窗⼝形状的卷积层和最⼤汇聚层来并⾏抽取信息，并使⽤1×1卷积层减少每像素级别上的通道维数从⽽降低模型复杂度。

2.  GoogLeNet将多个设计精细的Inception块与其他层（卷积层、全连接层）串联起来。其中Inception块的通道数分配之⽐是在ImageNet数据集上通过⼤量的实验得来的。

3. GoogLeNet和它的后继者们⼀度是ImageNet上最有效的模型之⼀：它以较低的计算复杂度提供了类似的测试精度。



### Chapter 7.5 批量规范化(BatchNormalizaton)

批量规范化（batch normalization），这是⼀种流⾏且有效的技术，可持续加速深层⽹络的收敛速度。

再结合残差块，批量规范化使得研究⼈员能够训练100层以上的⽹络。



批量规范化应⽤于单个可选层（也可以应⽤到所有层），其原理如下：在每次训练迭代中，我们⾸先规范化输⼊，即通过减去其均值并除以其标准差，其中两者均基于当前⼩批量处理。接下来，我们应⽤⽐例系数和⽐例偏移。正是由于这个基于批量统计的标准化，才有了批量规范化的名称。

请注意，如果我们尝试使⽤⼤⼩为1的⼩批量应⽤批量规范化，我们将⽆法学到任何东西。这是因为在减去均值之后，每个隐藏单元将为0。所以，只有使⽤⾜够⼤的⼩批量，批量规范化这种⽅法才是有效且稳定的。请注意，在应⽤批量规范化时，批量⼤⼩的选择可能⽐没有批量规范化时更重要。

请注意，我们在⽅差估计值中添加⼀个⼩的常量*ϵ >* 0，以确保我们永远不会尝试除以零。批量规范化层在”训练模式“（通过⼩批量统计数据规范化）和“预测模式”（通过数据集统计规范化）中的功能不同。在训练过程中，我们⽆法得知使⽤整个数据集来估计平均值和⽅差，所以只能根据每个⼩批次的平均值和⽅差不断训练模型。⽽在预测模式下，可以根据整个数据集精确计算批量规范化所需的平均值和⽅差。

批量规范化和其他层之间的⼀个关键区别是，由于批量规范化在完整的⼩批量上运⾏，因此我们不能像以前在引⼊其他层时那样忽略批量⼤⼩。我们在下⾯讨论这两种情况：全连接层和卷积层，他们的批量规范化实现略有不同。



对于全连接层：

我们在特征维度axis=0上面做批量规范化。

对于卷积层：

卷积层的数据可以表示为四维矩阵(批量大小，通道，特征图长，特征图宽)，我们在axis=2也就是通道上做批量规范化，也就是说一个通道的所有批量进行平均值和方差的计算。因此，在计算平均值和⽅差时，我们会收集所有空间位置的值，然后在给定通道内应⽤相同的均值和⽅差，以便在每个空间位置对值进⾏规范化。

举个例子：

如果 min-batch sizes 为 m，那么网络某一层输入数据可以表示为四维矩阵(m,f,w,h)，m 为 min-batch sizes，f 为特征图个数，w、h 分别为特征图的宽高。在 CNN 中我们可以把每个特征图看成是一个特征处理（一个神经元），因此在使用 Batch Normalization，mini-batch size 的大小就是：m*w*h，于是对于每个特征图都只有一对可学习参数：γ、β。

我们可以调用nn库函数来方便实现BN：

```python3
nn.BatchNorm2d(上一个卷积层的输出通道数通道数)
# 实现卷积层的BN
nn.BatchNorm1d(上一个fc的输出特征数)
# 实现fc的BN
```

总结:

1. 在模型训练过程中，批量规范化利⽤⼩批量的均值和标准差，不断调整神经⽹络的中间输出，使整个神

经⽹络各层的中间输出值更加稳定。

2. 批量规范化在全连接层和卷积层的使⽤略有不同。

3. 批量规范化层和暂退层⼀样，在训练模式和预测模式下计算不同。

4. 批量规范化有许多有益的副作⽤，主要是正则化。另⼀⽅⾯，”减少内部协变量偏移“的原始动机似乎不是⼀个有效的解释。



### Chapter 7.6 残差网络(Resnet)

一般来说，神经网络的深度不应该太深，否则会出现梯度消失和梯度爆炸的情况。loss值在减小到一定程度后，随着训练的增加甚至会反弹。

对于误差的链式反向传播，一旦其中某一个导数很小，多次连乘后梯度可能越来越小。对于深层网络，梯度传到浅层几乎就没了。

使用残差网络，相当于每一个导数加上了一个恒等项1。此时，就算原来的导数很小，误差仍然能够有效的反向传播。



残差⽹络的核⼼思想是：每个附加层都应该更容易地包含原始函数作为其元素之⼀。



ResNet沿⽤了VGG完整的3 *×* 3卷积层设计。残差块⾥⾸先有2个有相同输出通道数的3 *×* 3卷积层。每个卷积层后接⼀个批量规范化层和ReLU激活函数。然后我们通过跨层数据通路，跳过这2个卷积运算，将输⼊直接加在最后的ReLU激活函数前。这样的设计要求2个卷积层的输出与输⼊形状⼀样，从⽽使它们可以相加。如果想改变通道数，就需要引⼊⼀个额外的1 *×* 1卷积层来将输⼊变换成需要的形状后再做相加运算。

```python3
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
class Residual(nn.Module): #@save
	def __init__(self, input_channels, num_channels,
		use_1x1conv=False, strides=1):
        super().__init__()
		self.conv1 = nn.Conv2d(input_channels, num_channels,kernel_size=3, padding=1, stride=strides)
		self.conv2 = nn.Conv2d(num_channels, num_channels,kernel_size=3, padding=1)
		if use_1x1conv:
			self.conv3 = nn.Conv2d(input_channels, num_channels,kernel_size=1, stride=strides)
		else:
			self.conv3 = None
			self.bn1 = nn.BatchNorm2d(num_channels)
			self.bn2 = nn.BatchNorm2d(num_channels)
	def forward(self, X):
		Y = F.relu(self.bn1(self.conv1(X)))
		Y = self.bn2(self.conv2(Y))
		if self.conv3:
			X = self.conv3(X)
		Y += X
		return F.relu(Y)

```

此代码⽣成两种类型的⽹络：⼀种是当use_1x1conv=False时，应⽤ReLU⾮线性函数之前，将输⼊添加到输出。另⼀种是当use_1x1conv=True时，添加通过1 *×* 1卷积调整通道和分辨率。



ResNet的前两层跟之前介绍的GoogLeNet中的⼀样：在输出通道数为64、步幅为2的7 *×* 7卷积层后，接步幅为2的3 *×* 3的最⼤汇聚层。不同之处在于ResNet每个卷积层后增加了批量规范化层。

```python3
b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),nn.BatchNorm2d(64), nn.ReLU(),
nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
```

GoogLeNet在后⾯接了4个由Inception块组成的模块。ResNet则使⽤4个由残差块组成的模块，每个模块使⽤若⼲个同样输出通道数的残差块。第⼀个模块的通道数同输⼊通道数⼀致。由于之前已经使⽤了步幅为2的最⼤汇聚层，所以⽆须减⼩⾼和宽。之后的每个模块在第⼀个残差块⾥将上⼀个模块的通道数翻倍，并将⾼和宽减半。

```python3
def resnet_block(input_channels, num_channels, num_residuals,
	first_block=False):
	blk = []
	for i in range(num_residuals):
		if i == 0 and not first_block:
			blk.append(Residual(input_channels, num_channels,
			use_1x1conv=True, strides=2))
		else:
			blk.append(Residual(num_channels, num_channels))
	return blk
```

接着在ResNet加⼊所有残差块，这⾥每个模块使⽤2个残差块。

```python3
b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
b3 = nn.Sequential(*resnet_block(64, 128, 2))
b4 = nn.Sequential(*resnet_block(128, 256, 2))
b5 = nn.Sequential(*resnet_block(256, 512, 2))
```

最后，与GoogLeNet⼀样，在ResNet中加⼊全局平均汇聚层，以及全连接层输出。

```python3
net = nn.Sequential(b1, b2, b3, b4, b5,nn.AdaptiveAvgPool2d((1,1)),nn.Flatten(), nn.Linear(512, 10))
```

每个模块有4个卷积层（不包括恒等映射的1 *×* 1卷积层）。加上第⼀个7 *×* 7卷积层和最后⼀个全连接层，共有18层。因此，这种模型通常被称为ResNet-18。



总结

1. 学习嵌套函数（nested function）是训练神经⽹络的理想情况。在深层神经⽹络中，学习另⼀层作为恒等映射（identity function）较容易（尽管这是⼀个极端情况）。

2. 残差映射可以更容易地学习同⼀函数，例如将权重层中的参数近似为零。

3. 利⽤残差块（residual blocks）可以训练出⼀个有效的深层神经⽹络：输⼊可以通过层间的残余连接更快地向前传播。

4. 残差⽹络（ResNet）对随后的深层神经⽹络设计产⽣了深远影响。

### Chapter 7.7 稠密连接网络(DenseNet)

ResNet极⼤地改变了如何参数化深层⽹络中函数的观点。稠密连接⽹络（DenseNet）在某种程度上是ResNet的逻辑扩展。

ResNet和DenseNet的关键区别在于，DenseNet输出是连接⽽不是如ResNet的简单相加。

DenseNet实现起来⾮常简单：我们不需要添加术语，⽽是将它们连接起来。DenseNet这个名字由变量之间的“稠密连接”⽽得来，最后⼀层与之前的所有层紧密相连。

稠密⽹络主要由2部分构成：稠密块（dense block）和过渡层（transition layer）。前者定义如何连接输⼊和输出，⽽后者则控制通道数量，使其不会太复杂。

DenseNet使⽤了ResNet改良版的“批量规范化、激活和卷积”架构。

```python3
def conv_block(input_channels, num_channels):
	return nn.Sequential(nn.BatchNorm2d(input_channels), nn.ReLU(),nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1))
```

**稠密块**

⼀个稠密块由多个卷积块组成，每个卷积块使⽤相同数量的输出通道。然⽽，在前向传播中，我们将每个卷积块的输⼊和输出在通道维上连结。

```python3
class DenseBlock(nn.Module):
	def __init__(self, num_convs, input_channels, num_channels):
		super(DenseBlock, self).__init__()
		layer = []
		for i in range(num_convs):
			layer.append(conv_block(num_channels * i + input_channels, num_channels))
		self.net = nn.Sequential(*layer)
	def forward(self, X):
		for blk in self.net:
			Y = blk(X)
			# 连接通道维度上每个块的输⼊和输出
			X = torch.cat((X, Y), dim=1)
		return X
```

**过渡层**

由于每个稠密块都会带来通道数的增加，使⽤过多则会过于复杂化模型。⽽过渡层可以⽤来控制模型复杂度。它通过1 *×* 1卷积层来减⼩通道数，并使⽤步幅为2的平均汇聚层减半⾼和宽，从⽽进⼀步降低模型复杂度。

```python3
def transition_block(input_channels, num_channels):
	return nn.Sequential(nn.BatchNorm2d(input_channels), 
                         nn.ReLU(),
                         nn.Conv2d(input_channels, num_channels, kernel_size=1),
                         nn.AvgPool2d(kernel_size=2, stride=2))
```

**DenseNet模型**

我们来构造DenseNet模型。DenseNet⾸先使⽤同ResNet⼀样的单卷积层和最⼤汇聚层。

```python3
b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                   nn.BatchNorm2d(64),
                   nn.ReLU(),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
```

接下来，类似于ResNet使⽤的4个残差块，DenseNet使⽤的是4个稠密块。与ResNet类似，我们可以设置每个稠密块使⽤多少个卷积层。这⾥我们设成4，从⽽与 ResNet-18保持⼀致。稠密块⾥的卷积层通道数（即增⻓率）设为32，所以每个稠密块将增加128个通道。

在每个模块之间，ResNet通过步幅为2的残差块减⼩⾼和宽，DenseNet则使⽤过渡层来减半⾼和宽，并减半通道数。

```python3
# num_channels为当前的通道数
num_channels, growth_rate = 64, 32
num_convs_in_dense_blocks = [4, 4, 4, 4]
blks = []
for i, num_convs in enumerate(num_convs_in_dense_blocks):
	blks.append(DenseBlock(num_convs, num_channels, growth_rate))
	# 上⼀个稠密块的输出通道数
	num_channels += num_convs * growth_rate
	# 在稠密块之间添加⼀个转换层，使通道数量减半
	if i != len(num_convs_in_dense_blocks) - 1:
		blks.append(transition_block(num_channels, num_channels // 2))
		num_channels = num_channels // 2
```

与ResNet类似，最后接上全局汇聚层和全连接层来输出结果。

```python3
net = nn.Sequential(b1, 
                    *blks,
                    nn.BatchNorm2d(num_channels), 
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(),
                    nn.Linear(num_channels, 10))
```



总结:

1. 在跨层连接上，不同于ResNet中将输⼊与输出相加，稠密连接⽹络（DenseNet）在通道维上连结输⼊与输出。

2. DenseNet的主要构建模块是稠密块和过渡层。

3. 在构建DenseNet时，我们需要通过添加过渡层来控制⽹络的维数，从⽽再次减少通道的数量。

## Chapter 13 计算机视觉

### Chapter 13.1 图像增广

⼤型数据集是成功应⽤深度神经⽹络的先决条件。图像增⼴在对训练图像进⾏⼀系列的随机变化之后，⽣成相似但不同的训练样本，从⽽扩⼤了训练集的规模。此外，应⽤图像增⼴的原因是，随机改变训练样本可以减少模型对某些属性的依赖，从⽽提⾼模型的泛化能⼒。例如，我们可以以不同的⽅式裁剪图像，使感兴趣的对象出现在不同的位置，减少模型对于对象出现位置的依赖。我们还可以调整亮度、颜⾊等因素来降低模型对颜⾊的敏感度。

### Chapter 13.2 微调

微调一般包含下述四个步骤:

1. 在源数据集上预训练神经⽹络模型，即源模型。

2. 创建⼀个新的神经⽹络模型，即⽬标模型。这将复制源模型上的所有模型设计及其参数（输出层除外）。我们假定这些模型参数包含从源数据集中学到的知识，这些知识也将适⽤于⽬标数据集。我们还假设源模型的输出层与源数据集的标签密切相关；因此不在⽬标模型中使⽤该层。

3. 向⽬标模型添加输出层，其输出数是⽬标数据集中的类别数。然后随机初始化该层的模型参数。

4. 在⽬标数据集（如椅⼦数据集）上训练⽬标模型。输出层将从头开始进⾏训练，⽽所有其他层的参数将根据源模型的参数进⾏微调。



总结:

1. 迁移学习将从源数据集中学到的知识“迁移”到⽬标数据集，微调是迁移学习的常⻅技巧。

2. 除输出层外，⽬标模型从源模型中复制所有模型设计及其参数，并根据⽬标数据集对这些参数进⾏微调。但是，⽬标模型的输出层需要从头开始训练。

3. 通常，微调参数使⽤较⼩的学习率，⽽从头开始训练输出层可以使⽤更⼤的学习率。

### Chapter 13.3 目标检测和边界框

在图像分类任务中，我们假设图像中只有⼀个主要物体对象，我们只关注如何识别其类别。然⽽，很多时候图像⾥有多个我们感兴趣的⽬标，我们不仅想知道它们的类别，还想得到它们在图像中的具体位置。在计算机视觉⾥，我们将这类任务称为⽬标检测（object detection）或⽬标识别（object recognition）。



总结:

1. ⽬标检测不仅可以识别图像中所有感兴趣的物体，还能识别它们的位置，该位置通常由矩形边界框表⽰。

2. 我们可以在两种常⽤的边界框表⽰（中间，宽度，⾼度）和（左上，右下）坐标之间进⾏转换。

### Chapter 13.4 锚框

⽬标检测算法通常会在输⼊图像中采样⼤量的区域，然后判断这些区域中是否包含我们感兴趣的⽬标，并调整区域边界从⽽更准确地预测⽬标的真实边界框（ground-truth bounding box）。不同的模型使⽤的区域采样⽅法可能不同。这⾥我们介绍其中的⼀种⽅法：以每个像素为中⼼，⽣成多个缩放⽐和宽⾼⽐（aspect ratio）不同的边界框。这些边界框被称为锚框（anchor box）。

假设输⼊图像的⾼度为*h*，宽度为*w*。我们以图像的每个像素为中⼼⽣成不同形状的锚框：缩放⽐为*s* *∈* (0*,* 1]，宽⾼⽐为*r >* 0。那么锚框的宽度和⾼度分别是*ws**√**r*和*hs*/*√* *r*。请注意，当中⼼位置给定时，已知宽和⾼的锚框是确定的。

要⽣成多个不同形状的锚框，让我们设置许多缩放⽐（scale）取值*s*1*, . . . , s* *n*和许多宽⾼⽐（aspect ratio）取值*r*1*, . . . , r* *m*。当使⽤这些⽐例和⻓宽⽐的所有组合以每个像素为中⼼时，输⼊图像将总共有*whnm*个锚框。尽管这些锚框可能会覆盖所有真实边界框，但计算复杂性很容易过⾼。在实践中，我们只考虑包含*s*1或*r*1的组合：(*s*1*, r*1)*,*(*s*1*, r*2)*, . . . ,*(*s*1*, r* *m*)*,*(*s*2*, r*1)*,*(*s*3*, r*1)*, . . . ,*(*s* *n* *, r*1)。

也就是说，以同⼀像素为中⼼的锚框的数量是*n* + *m* *−* 1。对于整个输⼊图像，我们将共⽣成*wh*(*n* + *m* *−* 1)个锚框。

我们刚刚提到某个锚框“较好地”覆盖了图像中的狗。如果已知⽬标的真实边界框，那么这⾥的“好”该如何如何量化呢？直观地说，我们可以衡量锚框和真实边界框之间的相似性。我们知道杰卡德系数（Jaccard）可以衡量两组之间的相似性。给定集合*A*和*B*，他们的杰卡德系数是他们交集的⼤⼩除以他们并集的⼤⼩：

事实上，我们可以将任何边界框的像素区域视为⼀组像素。通过这种⽅式，我们可以通过其像素集的杰卡德系数来测量两个边界框的相似性。对于两个边界框，我们通常将它们的杰卡德系数称为交并⽐（intersection over union，IoU），即两个边界框相交⾯积与相并⾯积之⽐。交并⽐的取值范围在0和1之间：0表⽰两个边界框⽆重合像素，1表⽰两个边界框完全重合。

在训练集中，我们将每个锚框视为⼀个训练样本。为了训练⽬标检测模型，我们需要每个锚框的类别（class）和偏移量（offset）标签，其中前者是与锚框相关的对象的类别，后者是真实边界框相对于锚框的偏移量。

**将真实边界框分配给锚框**

给定图像，假设锚框是*A*1*, A*2*, . . . , A**n**a*，真实边界框是*B*1*, B*2*, . . . , B**n**b*，其中*n**a* *≥* *n**b*。让我们定义⼀个矩阵**X** *∈* R *n* *a* *×* *n* *b*，其中第*i*⾏、第*j*列的元素*x* *ij*是锚框*A* *i*和真实边界框*B* *j*的IoU。该算法包含以下步骤：

1. 在矩阵**X**中找到最⼤的元素，并将它的⾏索引和列索引分别表⽰为*i*1和*j*1。然后将真实边界框*B* *j*1分配给锚框*A* *i*1。这很直观，因为*A* *i*1和*B* *j*1是所有锚框和真实边界框配对中最相近的。在第⼀个分配完成后，丢弃矩阵中*i*1 th⾏和*j*1 th列中的所有元素。

2. 在矩阵**X**中找到剩余元素中最⼤的元素，并将它的⾏索引和列索引分别表⽰为*i*2和*j*2。我们将真实边界框*B* *j*2分配给锚框*A* *i*2，并丢弃矩阵中*i*2 th⾏和*j*2 th列中的所有元素。

3. 此时，矩阵**X**中两⾏和两列中的元素已被丢弃。我们继续，直到丢弃掉矩阵**X**中*n* *b*列中的所有元素。此时，我们已经为这*n* *b*个锚框各⾃分配了⼀个真实边界框。

4. 只遍历剩下的*n* *a* *−* *n* *b*个锚框。例如，给定任何锚框*A* *i*，在矩阵**X**的第*i* th⾏中找到与*A* *i*的IoU最⼤的真实边界框*B* *j*，只有当此IoU⼤于预定义的阈值时，才将*B* *j*分配给*A* *i*。

**标记类别和偏移量**

我们需要为每个锚框标记类别和偏移量。假设⼀个锚框*A*被分配了⼀个真实边界框*B*。⼀⽅⾯，锚框*A*的类别将被标记为与*B*相同。另⼀⽅⾯，锚框*A*的偏移量将根据*B*和*A*中⼼坐标的相对位置以及这两个框的相对⼤⼩进⾏标记。鉴于数据集内不同的框的位置和⼤⼩不同，我们可以对那些相对位置和⼤⼩应⽤变换，使其获得分布更均匀且易于拟合的偏移量。



如果⼀个锚框没有被分配真实边界框，我们只需将锚框的类别标记为“背景”（background）。背景类别的锚框通常被称为“负类”锚框，其余的被称为“正类”锚框。

由于我们不关⼼对背景的检测，负类的偏移量不应影响⽬标函数。通过元素乘法，掩码变量中的零将在计算⽬标函数之前过滤掉负类偏移量。

**使⽤⾮极⼤值抑制预测边界框**

当有许多锚框时，可能会输出许多相似的具有明显重叠的预测边界框，都围绕着同⼀⽬标。为了简化输出，我们可以使⽤⾮极⼤值抑制（non-maximum suppression，NMS）合并属于同⼀⽬标的类似的预测边界框。以下是⾮极⼤值抑制的⼯作原理。对于⼀个预测边界框*B*，⽬标检测模型会计算每个类别的预测概率。假设最⼤的预测概率为*p*，则该概率所对应的类别*B*即为预测的类别。具体来说，我们将*p*称为预测边界框*B*的置信度（confidence）。在同⼀张图像中，所有预测的⾮背景边界框都按置信度降序排序，以⽣成列表*L*。然后我们通过以下步骤操作排序列表*L*：

1. 从*L*中选取置信度最⾼的预测边界框*B*1作为基准，然后将所有与*B*1的IoU超过预定阈值*ϵ*的⾮基准预测边界框从*L*中移除。这时，*L*保留了置信度最⾼的预测边界框，去除了与其太过相似的其他预测边界框。简⽽⾔之，那些具有⾮极⼤值置信度的边界框被抑制了。

2. 从*L*中选取置信度第⼆⾼的预测边界框*B*2作为⼜⼀个基准，然后将所有与*B*2的IoU⼤于*ϵ*的⾮基准预测边界框从*L*中移除。

3. 重复上述过程，直到*L*中的所有预测边界框都曾被⽤作基准。此时，*L*中任意⼀对预测边界框的IoU都⼩于阈值*ϵ*；因此，没有⼀对边界框过于相似。

4. 输出列表*L*中的所有预测边界框



总结:

1. 我们以图像的每个像素为中⼼⽣成不同形状的锚框。

2. 交并⽐（IoU）也被称为杰卡德系数，⽤于衡量两个边界框的相似性。它是相交⾯积与相并⾯积的⽐率。

3. 在训练集中，我们需要给每个锚框两种类型的标签。⼀个是与锚框中⽬标检测的类别，另⼀个是锚框真实相对于边界框的偏移量。

4. 在预测期间，我们可以使⽤⾮极⼤值抑制（NMS）来移除类似的预测边界框，从⽽简化输出。

### Chapter 13.5 多尺度目标检测

锚框代表了图像不同区域的样本。然⽽，如果为每个像素都⽣成的锚框，我们最终可能会得到太多需要计算的锚框。想象⼀个561*×*728的输⼊图像，如果以每个像素为中⼼⽣成五个形状不同的锚框，就需要在图像上标记和预测超过200万个锚框（561 *×* 728 *×* 5）。

我们可以在输⼊图像中均匀采样⼀⼩部分像素，并以它们为中⼼⽣成锚框。此外，在不同尺度下，我们可以⽣成不同数量和不同⼤⼩的锚框。当使⽤较⼩的锚框检测较⼩的物体时，我们可以采样更多的区域，⽽对于较⼤的物体，我们可以采样较少的区域。



我们将卷积图层的⼆维数组输出称为特征图。通过定义特征图的形状，我们可以确定任何图像上均匀采样锚框的中⼼。

我们在特征图（fmap）上⽣成锚框（anchors），每个单位（像素）作为锚框的中⼼。由于锚框中的(*x, y*)轴坐标值（anchors）已经被除以特征图（fmap）的宽度和⾼度，因此这些值介于0和1之间，表⽰特征图中锚框的相对位置。

由于锚框（anchors）的中⼼分布于特征图（fmap）上的所有单位，因此这些中⼼必须根据其相对空间位置在任何输⼊图像上均匀分布。更具体地说，给定特征图的宽度和⾼度fmap_w和fmap_h，将均匀地对任何输⼊图像中fmap_h⾏和fmap_w列中的像素进⾏采样。以这些均匀采样的像素为中⼼，将会⽣成⼤⼩为s（假设列表s的⻓度为1）且宽⾼⽐（ratios）不同的锚框。



既然我们已经⽣成了多尺度的锚框，我们就将使⽤它们来检测不同尺度下各种⼤⼩的⽬标。

在某种规模上，假设我们有*c*张形状为*h* *×* *w*的特征图。我们⽣成了*hw*组锚框，其中每组都有*a*个中⼼相同的锚框。例如，给定10个（通道数量）4 *×* 4的特征图，我们⽣成了16组锚框，每组包含3个中⼼相同的锚框。接下来，每个锚框都根据真实值边界框来标记了类和偏移量。在当前尺度下，⽬标检测模型需要预测输⼊图像上*hw*组锚框类别和偏移量，其中不同组锚框具有不同的中⼼。

假设此处的*c*张特征图是CNN基于输⼊图像的正向传播算法获得的中间输出。既然每张特征图上都有*hw*个不同的空间位置，那么相同空间位置可以看作含有*c*个单元。根据 6.2节中对感受野的定义，特征图在相同空间位置的*c*个单元在输⼊图像上的感受野相同：它们表征了同⼀感受野内的输⼊图像信息。因此，我们可以将特征图在同⼀空间位置的*c*个单元变换为使⽤此空间位置⽣成的*a*个锚框类别和偏移量。本质上，我们⽤输⼊图像在某个感受野区域内的信息，来预测输⼊图像上与该区域位置相近的锚框类别和偏移量。

当不同层的特征图在输⼊图像上分别拥有不同⼤⼩的感受野时，它们可以⽤于检测不同⼤⼩的⽬标。例如，我们可以设计⼀个神经⽹络，其中靠近输出层的特征图单元具有更宽的感受野，这样它们就可以从输⼊图像中检测到较⼤的⽬标。

简⾔之，我们可以利⽤深层神经⽹络在多个层次上对图像进⾏分层表⽰，从⽽实现多尺度⽬标检测。



总结:

1. 在多个尺度下，我们可以⽣成不同尺⼨的锚框来检测不同尺⼨的⽬标。

2. 通过定义特征图的形状，我们可以决定任何图像上均匀采样的锚框的中⼼。

3. 我们使⽤输⼊图像在某个感受野区域内的信息，来预测输⼊图像上与该区域位置相近的锚框类别和偏移量。

4. 我们可以通过深⼊学习，在多个层次上的图像分层表⽰进⾏多尺度⽬标检测。

### Chapter 13.6 目标检测数据集

### Chapter 13.7 单发多框检测（SSD)



### Chapter 13.8 区域卷积神经网络

### Chapter 13.9 语义分割和数据集

### Chapter 13.10 转置卷积

### Chapter 13.11 全卷积网络

### Chapter 13.12 风格迁移



