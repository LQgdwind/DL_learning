import numpy as np
import torch
import d2l.torch as d2l
from torch.utils import data
# 线性回归的简洁实现

true_w = torch.tensor([2, -3.4])
true_b = 4.2
features,labels = d2l.synthetic_data(true_w,true_b,1000)

def load_array(data_arrays,batch_size,is_train = True):
    """构造一个pytorch数据迭代器"""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

batch_size = 10
data_iter = load_array((features,labels),batch_size)

from torch import nn

net = nn.Sequential(nn.Linear(in_features=2, out_features=1))

# 在使⽤net之前，我们需要初始化模型参数。如在线性回归模型中的权重和偏置。
# 深度学习框架通常有预定义的⽅法来初始化参数。
# 在这⾥，我们指定每个权重参数应该从均值为0、标准差为0.01的正态分布中随机采样，偏置参数将初始化为零。
# 正如我们在构造nn.Linear时指定输⼊和输出尺⼨⼀样，现在我们能直接访问参数以设定它们的初始值。我
# 们通过net[0]选择⽹络中的第⼀个图层，然后使⽤weight.data和bias.data⽅法访问参数。
# 我们还可以使⽤替换⽅法normal_和fill_来重写参数值。

net[0].weight.data.normal_(0., 0.01)
net[0].bias.data.fill_(0)

loss = nn.MSELoss(reduction='mean')

trainer = torch.optim.SGD(net.parameters(),lr=0.03)

num_epochs = 3
for epoch in range(num_epochs):
    for x,y in data_iter:
        l = loss(net(x),y)
        trainer.zero_grad()
        l.sum().backward()
        trainer.step()
    l = loss(net(features),labels)
    print("epoch: {arg1}, loss: {arg2}".format(arg1=epoch+1,arg2=l.mean()))

