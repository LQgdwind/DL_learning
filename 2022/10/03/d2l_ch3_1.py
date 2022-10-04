import random

import torch
import numpy as np

# 线性回归的具体实现


def synthetic_data(w, b, num_examples):
    """生成y = Xw+b+噪声"""
    x = torch.normal(0, 1, (num_examples,len(w)))
    y = torch.matmul(x, w) + b
    y += torch.normal(0, 0.01, y.shape)
    print(y.shape)
    return x,y.reshape((-1, 1))

true_w = torch.tensor([2, -3.4])
true_b = 4.2
features,labels = synthetic_data(true_w,true_b,1000)

# print('features: {arg1}\n, labels: {arg2}'.format(arg1=features[0],arg2=labels[0]))

def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0,num_examples,batch_size):
        batch_indices = torch.tensor((indices[i:min(i+batch_size,num_examples)]))
        yield features[batch_indices],labels[batch_indices]

w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
b = torch.zeros(1,requires_grad=True)

def linreg(x,w,b):
    """线性回归模型"""
    return torch.matmul(x,w)+b

def squared_loss(y_hat,y):
    """均方损失"""
    return 0.5*(y_hat - y.reshape(y_hat.shape))**2

def sgd(params, lr, batch_size):
    """小批量随机梯度下降"""
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()

batch_size = 10
lr = 0.03
num_epochs = 3
net = linreg
loss = squared_loss

for epoch in range(num_epochs):
    for x,y in data_iter(batch_size,features,labels):
        l = loss(net(x,w,b), y)
        l.sum().backward()
        sgd([w,b],lr,batch_size)
    with torch.no_grad():
        train_l = loss(net(features,w,b),labels)
        print("epoch {arg1}, loss {arg2}".format(arg1=epoch+1,arg2=train_l.mean()))
