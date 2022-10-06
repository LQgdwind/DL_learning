# 从零开始实现暂退法

import numpy as np
from d2l import torch as d2l
import torch
import matplotlib.pyplot as plt
from torch import nn

# 定义dropout层
def dropout_layer(x, dropout):
    assert 0 <= dropout <= 1
    # dropout = 1 时，所有元素都被丢弃
    if dropout == 1:
        return torch.zeros_like(x)
    elif dropout == 0:
        return x
    mask = (torch.rand(x.shape)>dropout).float()
    return mask*x/(1.0-dropout)

num_inputs, num_outputs, num_hiddens1, num_hiddens2 = 784, 10, 256, 256

dropout1, dropout2 = 0.2, 0.5

class Net(nn.Module):
    def __init__(self,num_inputs, num_outputs, num_hiddens1, num_hiddens2, is_training = True):
        super(Net,self).__init__()
        self.num_inputs = num_inputs
        self.training = is_training
        self.fc1 = nn.Linear(in_features=num_inputs,out_features=num_hiddens1)
        self.fc2 = nn.Linear(in_features=num_hiddens1,out_features=num_hiddens2)
        self.fc3 = nn.Linear(in_features=num_hiddens2,out_features=num_outputs)
        self.relu = nn.ReLU()

    def forward(self,x):
        h1 = self.relu(self.fc1(x.reshape((-1,self.num_inputs))))
        # 只有在训练模型的时候才使用dropout
        if self.training == True:
            # 在第一个全连接层后面增加一个dropout层
            h1 = dropout_layer(h1,dropout1)
        h2 = self.relu(self.fc2(h1))
        if self.training == True:
            # 在第二个全连接层后面增加一个dropout层
            h2 = dropout_layer(h2,dropout2)
        out = self.fc3(h2)
        return out

net = Net(num_inputs,num_outputs,num_hiddens1,num_hiddens2)

# 训练和测试
num_epochs, lr, batch_size = 10, 0.5, 256
loss = nn.CrossEntropyLoss(reduction='none')
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
trainer = torch.optim.SGD(net.parameters(),lr=lr)
d2l.train_ch3(net,train_iter,test_iter,loss,num_epochs,trainer)
plt.show()
