import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from d2l import torch as d2l

# 数据集生成
n_train, n_test, num_inputs, batch_size = 20, 100, 200, 5
true_w, true_b = torch.ones((num_inputs, 1)) * 0.01 , 0.05
train_data = d2l.synthetic_data(true_w, true_b,n_train)
train_iter = d2l.load_array(train_data, batch_size)
test_data = d2l.synthetic_data(true_w,true_b,n_test)
test_iter = d2l.load_array(test_data, batch_size, is_train=False)

# 从零开始实现正则化技术

def init_params():
    """初始化模型参数"""
    w = torch.normal(0,1,size=(num_inputs,1),requires_grad=True)
    b = torch.zeros(1,requires_grad=True)
    return [w,b]

def l2_penalty(w):
    """定义L2范数惩罚"""
    return torch.sum(w.pow(2))/2

def train(lambd):
    w, b = init_params()
    net, loss = lambda  x: d2l.linreg(x,w,b), d2l.squared_loss
    num_epochs, lr= 100, 0.003
    animator = d2l.Animator(xlabel='epochs',
                            ylabel='loss',
                            yscale='log',
                            xlim=[5, num_epochs],
                            legend=['train','test'])
    for epoch in range(num_epochs):
        for x,y in train_iter:
            # 增加了L2范数惩罚项
            # 广播机制使得l2_penalty(w)成为一个长度为batch_size的向量
            l = loss(net(x),y) + lambd * l2_penalty(w)
            l.sum().backward()
            d2l.sgd([w,b],lr,batch_size)
        if (epoch+1) % 5 == 0:
            animator.add(epoch+1,(d2l.evaluate_loss(net,train_iter,loss),
                                  d2l.evaluate_loss(net,test_iter,loss)))
    print('w的L2范数是: {arg1}'.format(arg1=torch.norm(w).item()))

# 0.5*λ*w^2为正则项，我们通过调整λ的大小来改变权重衰减的速度

# 忽略权重衰减
train(lambd=0)
plt.show()
# 令λ等于3
train(lambd=3)
plt.show()