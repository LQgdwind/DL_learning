import d2l.torch as d2l
import torch
import matplotlib.pyplot as plt


def show_heatmaps(matrices, xlabel, ylabel, titles=None, figsize=(2.5, 2.5),cmap='Reds'):
# """显⽰矩阵热图"""
    d2l.use_svg_display()
    num_rows, num_cols = matrices.shape[0], matrices.shape[1]
    fig, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize,sharex=True, sharey=True, squeeze=False)
    for i, (row_axes, row_matrices) in enumerate(zip(axes, matrices)):
        for j, (ax, matrix) in enumerate(zip(row_axes, row_matrices)):
            pcm = ax.imshow(matrix.detach().numpy(), cmap=cmap)
            if i == num_rows - 1:
                ax.set_xlabel(xlabel)
            if j == 0:
                ax.set_ylabel(ylabel)
            if titles:
                ax.set_title(titles[j])
    fig.colorbar(pcm, ax=axes, shrink=0.6)


attention_weights = torch.eye(10).reshape((1, 1, 10, 10))
show_heatmaps(attention_weights, xlabel='Keys', ylabel='Queries')
plt.show()

# 利用 yi = 2 sin(xi) + x0i .8 + ϵ生成一个⼈⼯数据集，
# 其中其中ϵ服从均值为0和标准差为0.5的正态分布

n_train = 50 # 训练样本数
x_train, _ = torch.sort(torch.rand(n_train) * 5) # 排序后的训练样本
# x_train为生成的数据tensor, _为排位的数字

def f(x):
    return 2 * torch.sin(x) + x**0.8


y_train = f(x_train) + torch.normal(0.0, 0.5, (n_train,)) # 训练样本的输出
x_test = torch.arange(0, 5, 0.1) # 测试样本
y_truth = f(x_test) # 测试样本的真实输出
n_test = len(x_test) # 测试样本数


def plot_kernel_reg(y_hat):
    d2l.plot(x_test, [y_truth, y_hat], 'x', 'y', legend=['Truth', 'Pred'], xlim=[0, 5], ylim=[-1, 5])
    d2l.plt.plot(x_train, y_train, 'o', alpha=0.5)

# 使用平均汇聚 ， 也就是 直接求平均值
y_hat = torch.repeat_interleave(y_train.mean(), n_test)
plot_kernel_reg(y_hat)
plt.show()

# NWK Nadaraya-Watson核回归（Nadaraya-Watson kernel regression）
# f(x) =∑α(x, xi)yi
# 其中x是查询，(xi, yi)是键值对。
# 注意⼒汇聚是yi的加权平均。
# 将查询x和键xi之间的关系建模为 注意⼒权重（attention weight）α(x, xi)

# Nadaraya-Watson核回归⾮参数模型:
# X_repeat的形状:(n_test,n_train),
# 每⼀⾏都包含着相同的测试输⼊（例如：同样的查询）
X_repeat = x_test.repeat_interleave(n_train).reshape((-1, n_train))
# x_train包含着键。attention_weights的形状：(n_test,n_train),
# 每⼀⾏都包含着要在给定的每个查询的值（y_train）之间分配的注意⼒权重
attention_weights = torch.nn.functional.softmax(-(X_repeat - x_train)**2 / 2, dim=1)
# 我们考虑⼀个⾼斯核（Gaussian kernel），其定义为：
# K(u) = 1/√2π*exp(−u^2/2 ).

# y_hat的每个元素都是值的加权平均值，其中的权重是注意⼒权重
y_hat = torch.matmul(attention_weights, y_train)
plot_kernel_reg(y_hat)
plt.show()
d2l.show_heatmaps(attention_weights.unsqueeze(0).unsqueeze(0),
                  xlabel='Sorted training inputs',
                  ylabel='Sorted testing inputs')
plt.show()

# 基于中的带参数的注意⼒汇聚，使⽤⼩批量矩阵乘法，定义Nadaraya-Watson核回归的带参数版本为
from torch import nn
class NWKernelRegression(nn.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.w = nn.Parameter(torch.rand((1,), requires_grad=True))
    def forward(self, queries, keys, values):
    # queries和attention_weights的形状为(查询个数，“键－值”对个数)
        queries = queries.repeat_interleave(keys.shape[1]).reshape((-1, keys.shape[1]))
        self.attention_weights = nn.functional.softmax(
        -((queries - keys) * self.w)**2 / 2, dim=1)
    # values的形状为(查询个数，“键－值”对个数)
        return torch.bmm(self.attention_weights.unsqueeze(1),values.unsqueeze(-1)).reshape(-1)

    # X_tile的形状:(n_train，n_train)，每⼀⾏都包含着相同的训练输⼊
    X_tile = x_train.repeat((n_train, 1))
    # Y_tile的形状:(n_train，n_train)，每⼀⾏都包含着相同的训练输出
    Y_tile = y_train.repeat((n_train, 1))
    # keys的形状:('n_train'，'n_train'-1)
    keys = X_tile[(1 - torch.eye(n_train)).type(torch.bool)].reshape((n_train, -1))
    # values的形状:('n_train'，'n_train'-1)
    values = Y_tile[(1 - torch.eye(n_train)).type(torch.bool)].reshape((n_train, -1))

# X_tile的形状:(n_train，n_train)，每⼀⾏都包含着相同的训练输⼊
X_tile = x_train.repeat((n_train, 1))
# Y_tile的形状:(n_train，n_train)，每⼀⾏都包含着相同的训练输出
Y_tile = y_train.repeat((n_train, 1))
# keys的形状:('n_train'，'n_train'-1)
keys = X_tile[(1 - torch.eye(n_train)).type(torch.bool)].reshape((n_train, -1))
# values的形状:('n_train'，'n_train'-1)
values = Y_tile[(1 - torch.eye(n_train)).type(torch.bool)].reshape((n_train, -1))

net = NWKernelRegression()
loss = nn.MSELoss(reduction='none')
trainer = torch.optim.SGD(net.parameters(), lr=0.5)
animator = d2l.Animator(xlabel='epoch', ylabel='loss', xlim=[1, 5])
for epoch in range(5):
    trainer.zero_grad()
    l = loss(net(x_train, keys, values), y_train)
    l.sum().backward()
    trainer.step()
    print(f'epoch {epoch + 1}, loss {float(l.sum()):.6f}')
    animator.add(epoch + 1, float(l.sum()))

# keys的形状:(n_test，n_train)，每⼀⾏包含着相同的训练输⼊（例如，相同的键）
keys = x_train.repeat((n_test, 1))
# value的形状:(n_test，n_train)
values = y_train.repeat((n_test, 1))
y_hat = net(x_test, keys, values).unsqueeze(1).detach()
plot_kernel_reg(y_hat)
plt.show()

# 与⾮参数的注意⼒汇聚模型相⽐，
# 带参数的模型加⼊可学习的参数后，
# 曲线在注意⼒权重较⼤的区域变得更不平滑

d2l.show_heatmaps(net.attention_weights.unsqueeze(0).unsqueeze(0),
                  xlabel='Sorted training inputs',
                  ylabel='Sorted testing inputs')
plt.show()

