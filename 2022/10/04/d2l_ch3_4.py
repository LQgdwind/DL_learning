import torch
from IPython import display
from d2l import torch as d2l
import matplotlib.pyplot as plt

batch_size = 256
train_iter,test_iter = d2l.load_data_fashion_mnist(batch_size)

num_inputs = 784
num_outputs = 10

w = torch.normal(0,0.01,size=(num_inputs,num_outputs),requires_grad=True)
b = torch.zeros(num_outputs,requires_grad=True)

def softmax(x):
    x_exp = torch.exp(x)
    partition = x_exp.sum(1,keepdim=True)
    return x_exp/partition

def net(x):
    return softmax(torch.matmul(x.reshape((-1,w.shape[0])),w)+b)

def cross_entropy(y_hat,y):
    return -torch.log(y_hat[range(len(y_hat)),y])

def accuracy(y_hat,y):
    """计算预测正确是数量"""
    if len(y_hat.shape)>1 and y_hat.shape[1]>1 :
        # 第一个条件是预测张量元素大于一，第二个条件是预测张量的类别大于1
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    # 先将预测值与标签值的数据类型转换成相同，然后在进行比较，返回一个布尔张量
    # 布尔张量在做求和的时候True为1,False为0
    return float(cmp.type(y.dtype).sum())

def evaluate_accuracy(net, data_iter):
    if isinstance(net,torch.nn.Module):
        net.eval()
    metric = Accumulator(2)
    with torch.no_grad():
        for x,y in data_iter:
            metric.add(accuracy(net(x),y),y.numel())
    return metric[0]/metric[1]

class Accumulator:
    """在n个变量上累加"""
    def __init__(self,n):
        self.data = [0.0]*n

    def add(self,*args):
        self.data = [a+float(b) for a,b in zip(self.data,args)]

    def reset(self):
        self.data = [0.0]*len(self.data)

    def __getitem__(self, item):
        return self.data[item]

def train_epoch_ch3(net, train_iter, loss, updater):
    """训练模型一个迭代周期"""
    # 将模型设置为训练模式
    if isinstance(net,torch.nn.Module):
        net.train()
    # 训练损失总和、训练准确度总和、样本数
    metric = Accumulator(3)
    for x,y in train_iter:
        y_hat = net(x)
        l = loss(y_hat,y)
        # 计算梯度并且更新参数
        if isinstance(updater,torch.optim.Optimizer):
            # 使用pytorch内置的优化器和损失函数
            updater.zero_grad()
            l.mean().backward()
            updater.step()
        else:
            # 使用定制的优化器和损失函数
            l.sum().backward()
            updater(x.shape[0])
        metric.add(float(l.sum()),accuracy(y_hat,y),y.numel())
    return metric[0]/metric[2],metric[1]/metric[2]

def train_ch3(net,train_iter,test_iter,loss,num_epochs,updater):
    """训练模型"""
    animator = d2l.Animator(xlabel='epoch',
                            xlim=[1,num_epochs],
                            ylim=[0.3, 0.9],
                            legend=['train loss','train acc','test acc'])
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net,train_iter,loss,updater)
        test_acc = evaluate_accuracy(net,test_iter)
        animator.add(epoch+1,train_metrics+(test_acc,))
    train_loss,train_acc = train_metrics

lr = 0.1
def updater(batch_size):
    return d2l.sgd([w,b],lr,batch_size)

num_epochs = 10
train_ch3(net,train_iter,test_iter,cross_entropy,num_epochs,updater)
plt.show()

def predict_ch3(net, test_iter,n=6):
    """预测标签"""
    for x,y in test_iter:
        break
    trues = d2l.get_fashion_mnist_labels(y)
    preds = d2l.get_fashion_mnist_labels(net(x).argmax(axis=1))
    titles = [true + '\n' + pred for true,pred in zip(trues,preds)]
    d2l.show_images(
        x[0:n].reshape((n,28,28)),1,n,titles = titles[0:n]
    )

predict_ch3(net,test_iter)
plt.show()