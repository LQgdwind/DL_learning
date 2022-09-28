import math
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
import matplotlib.pyplot as plt

# 隐藏层: H_t = ϕ(X_t*W_xh + H_t−1*W_hh + b_h).
# ϕ多用tanh
# 输出层: O_t = H_t*W_hq + b_q

# 困惑度 = ⼀个序列中所有的n个词元的交叉熵损失的平均值的指数
# exp(− 1/n*∑log P(x_t | x_t−1,..., x_1))

# 困惑度的最好的理解是“下⼀个词元的实际选择数的调和平均数”。我们看看⼀些案例：
# • 在最好的情况下，模型总是完美地估计标签词元的概率为1。在这种情况下，模型的困惑度为1。
# • 在最坏的情况下，模型总是预测标签词元的概率为0。在这种情况下，困惑度是正⽆穷⼤。
# • 在基线上，该模型的预测是词表的所有可⽤词元上的均匀分布。在这种情况下，困惑度等于词表中唯⼀词元的数量。
# 事实上，如果我们在没有任何压缩的情况下存储序列，这将是我们能做的最好的编码⽅式。
# 因此，这种⽅式提供了⼀个重要的上限，⽽任何实际模型都必须超越这个上限。

batch_size, num_steps = 32, 35
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)

def get_params(vocab_size, num_hiddens, device):
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        return torch.randn(size=shape, device=device) * 0.01

    # 隐藏层参数
    W_xh = normal((num_inputs, num_hiddens))
    W_hh = normal((num_hiddens, num_hiddens))
    b_h = torch.zeros(num_hiddens, device=device)
    # 输出层参数
    W_hq = normal((num_hiddens, num_outputs))
    b_q = torch.zeros(num_outputs, device=device)
    # 附加梯度
    params = [W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        param.requires_grad_(True)
    return params


def init_rnn_state(batch_size, num_hiddens, device):
    return (torch.zeros((batch_size, num_hiddens), device=device), )


def rnn(inputs, state, params):
# inputs的形状：(时间步数量，批量⼤⼩，词表⼤⼩)
    W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
# X的形状：(批量⼤⼩，词表⼤⼩)
    for X in inputs:
        H = torch.tanh(torch.mm(X, W_xh) + torch.mm(H, W_hh) + b_h)
        Y = torch.mm(H, W_hq) + b_q
        outputs.append(Y)
    return torch.cat(outputs, dim=0), (H,)

# 定义了所有需要的函数之后，
# 接下来我们创建⼀个类来包装这些函数，
# 并存储从零开始实现的循环神经⽹络模型的参数。


class RNNModelScratch: #@save
# """从零开始实现的循环神经⽹络模型"""
    def __init__(self, vocab_size, num_hiddens, device,get_params, init_state, forward_fn):
        self.vocab_size, self.num_hiddens = vocab_size, num_hiddens
        self.params = get_params(vocab_size, num_hiddens, device)
        self.init_state, self.forward_fn = init_state, forward_fn
    def __call__(self, X, state):
        X = F.one_hot(X.T, self.vocab_size).type(torch.float32)
        return self.forward_fn(X, state, self.params)
    def begin_state(self, batch_size, device):
        return self.init_state(batch_size, self.num_hiddens, device)


def predict_ch8(prefix, num_preds, net, vocab, device):
    state = net.begin_state(batch_size=1, device=device)
    outputs = [vocab[prefix[0]]]
    get_input = lambda: torch.tensor([outputs[-1]], device=device).reshape((1, 1))
    for y in prefix[1:]:  # 预热期
        _, state = net(get_input(), state)
        outputs.append(vocab[y])
    for _ in range(num_preds):  # 预测num_preds步
        y, state = net(get_input(), state)
        outputs.append(int(y.argmax(dim=1).reshape(1)))
    return ''.join([vocab.idx_to_token[i] for i in outputs])


X = torch.arange(10).reshape((2, 5))
F.one_hot(X.T, 28).shape
num_hiddens = 512
net = RNNModelScratch(len(vocab), num_hiddens, d2l.try_gpu(), get_params,init_rnn_state, rnn)
state = net.begin_state(X.shape[0], d2l.try_gpu())
Y, new_state = net(X.to(d2l.try_gpu()), state)
print(Y.shape, len(new_state), new_state[0].shape)


str = predict_ch8('time traveller ', 10, net, vocab, d2l.try_gpu())
print(str)

# 我们知道梯度范数永远不会超过θ，并且更新后的梯度完全与g的原始⽅向对⻬。
# 它还有⼀个值得拥有的副作⽤，
# 即限制任何给定的⼩批量数据（以及其中任何给定的样本）对参数向量的影响，
# 这赋予了模型⼀定程度的稳定性。
# 梯度裁剪提供了⼀个快速修复梯度爆炸的⽅法，
# 虽然它并不能完全解决问题。
# 但它是众多有效的技术之⼀。
# g ← min (1, θ/∥g∥) g. ||g||为第二范数


# 梯度裁剪:
def grad_clipping(net, theta):
    if isinstance(net, nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm
# 梯度裁剪可以防⽌梯度爆炸，但不能应对梯度消失。


def train_epoch_ch8(net, train_iter, loss, updater, device, use_random_iter):
    state, timer = None, d2l.Timer()
    metric = d2l.Accumulator(2)  # 训练损失之和,词元数量
    for X, Y in train_iter:
        if state is None or use_random_iter:
         # 在第⼀次迭代或使⽤随机抽样时初始化state
            state = net.begin_state(batch_size=X.shape[0], device=device)
    else:
        if isinstance(net, nn.Module) and not isinstance(state, tuple):
        # state对于nn.GRU是个张量
            state.detach_()
        else:
        # state对于nn.LSTM或对于我们从零开始实现的模型是个张量
            for s in state:
                s.detach_()
    y = Y.T.reshape(-1)
    X, y = X.to(device), y.to(device)
    y_hat, state = net(X, state)
    l = loss(y_hat, y.long()).mean()
    if isinstance(updater, torch.optim.Optimizer):
        updater.zero_grad()
        l.backward()
        grad_clipping(net, 1)
        updater.step()
    else:
        l.backward()
        grad_clipping(net, 1)  # 因为已经调⽤了mean函数
        updater(batch_size=1)
    metric.add(l * y.numel(), y.numel())
    return math.exp(metric[0] / metric[1]), metric[1] / timer.stop()

def train_ch8(net, train_iter, vocab, lr, num_epochs, device,
              use_random_iter=False):
    loss = nn.CrossEntropyLoss()
    animator = d2l.Animator(xlabel='epoch', ylabel='perplexity',
                            legend=['train'], xlim=[10, num_epochs])
    # 初始化
    if isinstance(net, nn.Module):
        updater = torch.optim.SGD(net.parameters(), lr)
    else:
        updater = lambda batch_size: d2l.sgd(net.params, lr, batch_size)
    predict = lambda prefix: predict_ch8(prefix, 50, net, vocab, device)
    # 训练和预测
    for epoch in range(num_epochs):
        ppl, speed = train_epoch_ch8(
            net, train_iter, loss, updater, device, use_random_iter)
        if (epoch + 1) % 10 == 0:
            print(predict('time traveller'))
            animator.add(epoch + 1, [ppl])
    # print(f'困惑度 {ppl:.1f}, {speed:.1f} 词元/秒 {str(device)}')
    print(predict('time traveller'))
    print(predict('traveller'))


num_epochs, lr = 10000, 1
train_ch8(net, train_iter, vocab, lr, num_epochs, d2l.try_gpu())
plt.show()
