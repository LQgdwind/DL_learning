import torch
from d2l import torch as d2l
import matplotlib.pyplot as plt
import numpy as np
from torch import nn

# 梯度消失
x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
y = torch.sigmoid(x)
y.backward(torch.ones_like(x))

d2l.plot(x.detach().numpy(),[y.detach().numpy(),x.grad.numpy()],
         legend=['sigmoid', 'gradient'],
         figsize=(4.5, 2.5))
plt.show()

# 梯度爆炸
m = torch.normal(0, 1, size=(4, 4))
print('一个矩阵:\n {arg1}'.format(arg1=m))
for i in range(100):
    m = torch.mm(m,torch.normal(0,1,size=(4,4)))
print('乘以100个矩阵后:\n {arg1}'.format(arg1=m))
