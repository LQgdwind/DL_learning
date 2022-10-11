# Linear Regression
# 线性回归是利用数理统计中回归分析，来确定两种或两种以上变量间相互依赖的定量关系的一种统计分析方法，运用十分广泛。其表达形式为y = w'x+e，e为误差服从均值为0的正态分布。
# 回归分析中，只包括一个自变量和一个因变量，且二者的关系可用一条直线近似表示，这种回归分析称为一元线性回归分析。如果回归分析中包括两个或两个以上的自变量，且因变量和自变量之间是线性关系，则称为多元线性回归分析。
# 简单的说： 线性回归对于输入x与输出y有一个映射f，y=f(x),而f的形式为aX+b。其中a和b是两个可调的参数，我们训练的时候就是训练a，b这两个参数。

import torch
from torch.nn import Linear, Module, MSELoss
from torch.optim import SGD
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import time

x = np.linspace(start=0,
                stop=20,
                num=500)
# 生成等差数列
y = 5 * x + 7
plt.plot(x, y)
plt.show()

x = np.random.rand(256)
noise = np.random.randn(256)/4
y = x*5 + 7 + noise
df = pd.DataFrame()
df['x'] = x
df['y'] = y


learning_rate, epochs = 0.05, 400
model = Linear(in_features=1,
               out_features=1)
criterion = MSELoss()
trainer = SGD(model.parameters(),
              lr=learning_rate)
x_train = x.reshape(-1, 1).astype('float32')
y_train = y.reshape(-1, 1).astype('float32')
# 使用astype的原因是为了在下述训练过程中直接实现tensor转换。
for epoch in range(epochs):
    inputs = torch.from_numpy(x_train)
    labels = torch.from_numpy(y_train)
    outputs = model(inputs)
    loss = criterion(outputs,labels)
    trainer.zero_grad()
    loss.backward()
    trainer.step()
    if epoch % 2 == 0:
        print('epoch {},loss {:1.4f}'.format(epoch,loss.data.item()))
        predicted = model.forward(torch.from_numpy(x_train)).data.numpy()
        plt.plot(x_train, y_train, 'go', label='data', alpha=0.3)
        # 'go' 'g'表示绿色 'o'表示点形为圆
        plt.plot(x_train, predicted, label='predicted', alpha=1)
        plt.legend()
        # 显示图例，图例中的内容由label定义
        plt.show()
        time.sleep(0.5)


