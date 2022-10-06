# 暂退法的简洁实现

import torch
import matplotlib.pyplot as plt
import numpy as np
from d2l import torch as d2l
from torch import nn

dropout1, dropout2 = 0.2, 0.5
net = nn.Sequential(
    nn.Flatten(),
    nn.Linear(in_features=784,
              out_features=256),
    nn.ReLU(),
    nn.Dropout(p=dropout1),
    nn.Linear(in_features=256,
              out_features=256),
    nn.ReLU(),
    nn.Dropout(p=dropout2),
    nn.Linear(in_features=256,
              out_features=10))

def init_weights(m):
    if isinstance(m,nn.Linear):
        nn.init.normal_(m.weight, std=0.01)

net.apply(init_weights)

# 训练和测试
num_epochs, lr, batch_size = 10, 0.5, 256
loss = nn.CrossEntropyLoss(reduction='none')
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
trainer = torch.optim.SGD(net.parameters(),lr=lr)
d2l.train_ch3(net,train_iter,test_iter,loss,num_epochs,trainer)
plt.show()