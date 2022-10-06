import torch
import numpy as np
import matplotlib.pyplot as plt
from d2l import torch as d2l
import torchvision

# 数据集生成
n_train, n_test, num_inputs, batch_size = 20, 100, 200, 5
true_w, true_b = torch.ones((num_inputs, 1)) * 0.01 , 0.05
train_data = d2l.synthetic_data(true_w, true_b,n_train)
train_iter = d2l.load_array(train_data, batch_size)
test_data = d2l.synthetic_data(true_w,true_b,n_test)
test_iter = d2l.load_array(test_data, batch_size, is_train=False)

def train_concise(wd):
    net = torch.nn.Sequential(
        torch.nn.Linear(in_features=num_inputs,
                        out_features=1))
    for param in net.parameters():
        param.data.normal_()
    loss = torch.nn.MSELoss(reduction='none')
    num_epochs, lr =100,0.003
    # 偏置参数没有衰减
    trainer = torch.optim.SGD([
        {"params":net[0].weight,'weight_decay':wd},
        {"params":net[0].bias}],
    lr= lr)
    animator = d2l.Animator(xlabel='epochs',ylabel='loss',yscale='log',
                            xlim=[5, num_epochs],legend=['train','test'])
    for epoch in range(num_epochs):
        for x,y in train_iter :
            trainer.zero_grad()
            l = loss(net(x),y)
            l.mean().backward()
            trainer.step()
        if (epoch+1)%5 == 0:
            animator.add(epoch+1,
                         (d2l.evaluate_loss(net,train_iter,loss),
                          d2l.evaluate_loss(net,test_iter,loss)))
    print('w的L2范数: {arg1} '.format(arg1=net[0].weight.norm().item()))

train_concise(wd=0)
plt.show()
train_concise(wd=3)
plt.show()
