import torch
from torch import nn
from d2l import torch as d2l
import matplotlib.pyplot as plt
# 保留⼀些对过去观测的总结h_t，
# 并且同时更新预测xˆt和总结h_t。
# 这就产⽣了基于xˆt = P(x_t | h_t)估计x_t，
# 以及公式h_t = g(h_t−1, x_t−1)更新的模型。
# 由于h_t从未被观测到，
# 这类模型也被称为 隐变量⾃回归模型（latent autoregressive models）

# 在⾃回归模型的近似法中，
# 我们使⽤x_t−1,..., x_t−τ
# ⽽不是x_t−1,..., x_1来估计x_t。
# 只要这种是近似精确的，
# 我们就说序列满⾜⻢尔可夫条件（Markov condition）。
# 特别是，如果τ = 1，得到⼀个
# ⼀阶⻢尔可模型（first-order Markov model），
# P(x)由下式给出：
# P(x_1,..., x_T ) = ∏P(x_t | x_t−1)
# 当 P(x_1 | x_0) = P(x_1).

T = 1000 # 总共产⽣1000个点
time = torch.arange(1, T + 1, dtype=torch.float32)
x = torch.sin(0.01 * time) + torch.normal(0, 0.2, (T,))
# ：使⽤正弦函数和⼀些可加性噪声来⽣成序列数据，时间步为1, 2, . . . , 1000
d2l.plot(time, [x], 'time', 'x', xlim=[1, 1000], figsize=(6, 3))
plt.show()

tau = 4
features = torch.zeros((T - tau, tau))
for i in range(tau):
    features[:, i] = x[i: T - tau + i]
labels = x[tau:].reshape((-1, 1))

batch_size, n_train = 16, 600
# 只有前n_train个样本⽤于训练
train_iter = d2l.load_array((features[:n_train], labels[:n_train]),
                            batch_size, is_train=True)

# 初始化⽹络权重的函数
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
# ⼀个简单的多层感知机
def get_net():
    net = nn.Sequential(nn.Linear(4, 10),
    nn.ReLU(),
    nn.Linear(10, 1))
    net.apply(init_weights)
    return net
# 平⽅损失。注意：MSELoss计算平⽅误差时不带系数1/2
loss = nn.MSELoss(reduction='none')

def train(net, train_iter, loss, epochs, lr):
    trainer = torch.optim.Adam(net.parameters(), lr)
    for epoch in range(epochs):
        for X, y in train_iter:
            trainer.zero_grad()
            l = loss(net(X), y)
            l.sum().backward()
            trainer.step()
            print(f'epoch {epoch + 1}, ' f'loss: {d2l.evaluate_loss(net, train_iter, loss):f}')

net = get_net()
train(net, train_iter, loss, 5, 0.01)

# 下面的例子将展现K步预测的精度快速下降
max_steps = 64
features = torch.zeros((T - tau - max_steps + 1, tau + max_steps))
# 列i（i<tau）是来⾃x的观测，其时间步从（i+1）到（i+T-tau-max_steps+1）
for i in range(tau):
    features[:, i] = x[i: i + T - tau - max_steps + 1]
# 列i（i>=tau）是来⾃（i-tau+1）步的预测，其时间步从（i+1）到（i+T-tau-max_steps+1）
for i in range(tau, tau + max_steps):
    features[:, i] = net(features[:, i - tau:i]).reshape(-1)
steps = (1, 4, 16, 64)
d2l.plot([time[tau + i - 1: T - max_steps + i] for i in steps],
         [features[:, (tau + i - 1)].detach().numpy() for i in steps],
         'time',
         'x',
         legend=[f'{i}-step preds' for i in steps],
         xlim=[5, 1000],
         figsize=(6, 3))
plt.show()

