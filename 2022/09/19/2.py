import torch
import torch.nn as nn
import torch.nn.functional as F

# PyTorch中已经为我们准备好了现成的网络模型，
# 只要继承nn.Module，
# 并实现它的forward方法，
# PyTorch会根据autograd，
# 自动实现backward函数，
# 在forward函数中可使用任何tensor支持的函数，
# 还可以使用if、for循环、print、log等Python语法，
# 写法和标准的Python写法一致。
class Net(nn.Module):
    def __init__(self):
        #执行父类构造函数
        super(Net,self).__init__()
        #conv1是卷积层，fc1是全连接层
        self.conv1 = nn.Conv2d(in_channels=1,
                               out_channels=6,
                               kernel_size=(3, 3))
        self.fc1 = nn.Linear(in_features=1350,
                             out_features=10)

    #正向传播
    def forward(self,x):
        print(x.size())

        x = self.conv1(x)
        x = F.relu(x)

        print(x.size())

        x = F.max_pool2d(x, (2, 2))
        x = F.relu(x)
        print(x.size())
        x = x.reshape(x.size()[0], -1)
        # -1代表自适应，这里也可以用x.view()
        # 这里把四维的张量压缩成了两维

        print(x.size())

        x = self.fc1(x)

        return x


net = Net()
print(net)

# 网络的可学习参数通过net_parameters()返回
for parameters in net.parameters():
    print(parameters)

# net.named_parameters可同时返回可学习的参数及名称。

for name, parameters in net.named_parameters():
    print(name, ':', parameters.size())

# 随机初始化输入
input = torch.randn(1, 1, 32, 32)
out = net(input)
out.size()

# 在反向传播之前，先要将参数清零
net.zero_grad()
out.backward(torch.ones_like(out))

# 注意:torch.nn只支持mini-batches，
# 不支持一次只输入一个样本，即一次必须是一个batch。
# 也就是说，
# 就算我们输入一个样本，
# 也会对样本进行分批，
# 所以，所有的输入都会增加一个维度，
# 我们对比下刚才的input，nn中定义为3维，
# 但是我们人工创建时多增加了一个维度，
# 变为了4维，最前面的1即为batch-size

# 在nn中PyTorch还预制了常用的损失函数，
# 下面我们用MSELoss用来计算均方误差

y = torch.arange(0, 10).view(1, 10).float()
criterion = nn.MSELoss()
loss = criterion(out, y)
# loss 是一个标量,我们可以直接用item获取其数值
print(loss.item())


# 优化器
# 在反向传播计算完所有参数的梯度后，
# 还需要使用优化方法来更新网络的权重和参数，
# 例如随机梯度下降法(SGD)的更新策略如下：
# weight = weight - learning_rate * gradient

def main():
    learning_rate = 0.01
    label = torch.arange(0, 10).view(1, 10).float()
    out = net(input)
    criterion = nn.MSELoss(reduction='mean')
    # nn.MSELoss()参数介绍
    # (1)如果 reduction = ‘none’，直接返回向量形式的 loss
    # (2)如果 reduction ≠ ‘none’，那么 loss 返回的是标量
    # 　　a)如果 reduction=‘mean’，返回 loss.mean(); 注意：默认情况下， reduction=‘mean’
    # 　　b)如果 reduction=‘sum’，返回 loss.sum();
    loss = criterion(out, y)
    trainer = torch.optim.SGD(net.parameters(), lr=learning_rate)
    # 新建优化器
    trainer.zero_grad()
    # 梯度清零
    loss.backward()
    # 反向传播，求得梯度
    trainer.step()
    # 利用求得的梯度更新参数

# main函数为一次传播的模板
main()