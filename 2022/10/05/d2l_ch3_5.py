# softmax回归的简洁实现
import torch
from torch import nn
from d2l import torch as d2l
import matplotlib.pyplot as plt

batch_size = 256
train_iter,test_iter = d2l.load_data_fashion_mnist(batch_size)

net = nn.Sequential(nn.Flatten(),
                    nn.Linear(784,256),
                    nn.ReLU(),
                    nn.Linear(256,64),
                    nn.ReLU(),
                    nn.Linear(64,10)
                    )
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight,std=0.01)

net.apply(init_weights)

loss = nn.CrossEntropyLoss(reduction='none')

trainer = torch.optim.SGD(net.parameters(),lr=0.1)

def train_ch3(net,train_iter,test_iter,loss,num_epochs,updater):
    """训练模型"""
    animator = d2l.Animator(xlabel='epoch',
                            xlim=[1,num_epochs],
                            ylim=[0.3, 0.9],
                            legend=['train loss','train acc','test acc'])
    for epoch in range(num_epochs):
        train_metrics = d2l.train_epoch_ch3(net,train_iter,loss,updater)
        test_acc = d2l.evaluate_accuracy(net,test_iter)
        animator.add(epoch+1,train_metrics+(test_acc,))
    train_loss,train_acc = train_metrics

num_epochs = 10
train_ch3(net,train_iter,test_iter,loss,num_epochs,trainer)
plt.show()
