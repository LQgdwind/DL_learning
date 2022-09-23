# pytorch 中使用 nn.RNN 类来搭建基于序列的循环神经网络，它的构造函数有以下几个参数：
# - input_size：输入数据X的特征值的数目。
# - hidden_size：隐藏层的神经元数量，也就是隐藏层的特征数量。
# - num_layers：循环神经网络的层数，默认值是 1。
# - bias：默认为 True，如果为 false 则表示神经元不使用 bias 偏移参数。
# - batch_first：
#       如果设置为 True，则输入数据的维度中第一个维度就是 batch 值，
#       默认为 False。默认情况下第一个维度是序列的长度，第二个维度才是 - - batch，第三个维度是特征数目。
# - dropout：如果不为空，则表示最后跟一个 dropout 层抛弃部分数据，抛弃数据的比例由该参数指定。
import torch
input_tensor = torch.randn(100, 32, 20)
# 如果是句子，则这个input有32句话，每句话有100个单词，每个单词有50个特征。

# input = torch.randn(seq_len, batch_size, input_feature_size)
# # 这里非常的坑爹，因为batch_size 批量大小放在了第二个参数。。。
# 举个例子:
# 定义输入序列的长度 seq_len = 3
# 定义batch的长度 batch_size = 1
# 输入特征的数量/维 input_size  = 2
# 定义输入样本 input = torch.randn(3, 1, 2) print(input.shape) print(input)
# 备注：
# 输入是一个样本输入特征的长度为2， 样本的序列化串的长度为3。因此输入shape=3*2。
# 比如文本：“I love you”，就是一个序列，由三个单词组成，不管单词由多少个字母组成，每个单词被编码成一个长度=2的向量。

hidden_tensor_prev = torch.randn(10, 32, 50)
# hidden = torch.randn(hidden_layer_nums, batch_size, output_feature_size)
# 第一个参数是隐藏层的个数，第二个参数是批量大小，第三个参数是输出特征的数目。
net = torch.nn.RNN(input_size=20,
                   hidden_size=50,
                   num_layers=10)
output, hidden_tensor = net(input_tensor, hidden_tensor_prev)
print(output.size())
print(hidden_tensor.size())
# RNN 中最主要的参数是 input_size 和 hidden_size，这两个参数务必要搞清楚。其余的参数通常不用设置，采用默认值就可以了。


class RNN(torch.nn.Module):
    def __init__(self,input_size,hidden_size):
        super().__init__()
        self.W_x = torch.nn.Linear(in_features=input_size,
                                  out_features=hidden_size)
        self.W_h = torch.nn.Linear(in_features=hidden_size,
                                   out_features=hidden_size)

    def step(self,input_size,hidden_size):
        output = torch.tanh(self.W_h(hidden_size)+self.W_x(input_size))
        hidden_update = self.W_h.weight
        return output,hidden_update

    def __call__(self, input_size,hidden_size):
        return self.step(input_size,hidden_size)
    #   __call__方法可以让对象类似于函数一样被调用
    #   有点像cpp中重载类的()运算符。

# pytorch 使用torch.nn.lstm()实现长短记忆模型
# 输入的参数列表包括:
# input_size 输入数据的特征维数，通常就是embedding_dim(词向量的维度)
# hidden_size　LSTM中隐层的维度
# num_layers　循环神经网络的层数
# bias　用不用偏置，default=True
# batch_first 这个要注意，通常我们输入的数据shape=(batch_size,seq_length,embedding_dim),而batch_first默认是False,所以我们的输入数据最好送进LSTM之前将batch_size与seq_length这两个维度调换
# dropout　默认是0，代表不用dropout
# bidirectional默认是false，代表不用双向LSTM


# 输入数据
input_tensor = torch.randn(10,32,20)
hidden_tensor_prev = torch.randn(5,32,50)
cell_tensor_prev = torch.randn(5,32,50)

# 输入张量与状态张量与RNN的含义大致相同,cell与hidden的形状相同
# input就是shape=(seq_length,batch_size,input_size)的张量
# h_0是shape=(num_layers*num_directions,batch_size,hidden_size)的张量，它包含了在当前这个batch_size中每个句子的初始隐藏状态。其中num_layers就是LSTM的层数。如果bidirectional=True,num_directions=2,否则就是１，表示只有一个方向。
# c_0和h_0的形状相同，它包含的是在当前这个batch_size中的每个句子的初始细胞状态。h_0,c_0如果不提供，那么默认是０。

# 输出数据
# __call__方法有两个返回值:output,(h_n,c_n):
# output的shape=(seq_length,batch_size,num_directions*hidden_size),
# 它包含的是LSTM的最后一时间步的输出特征(h_t),ｔ是batch_size中每个句子的长度。
# h_n.shape==(num_directions * num_layers,batch,hidden_size)
# c_n.shape==h_n.shape
# h_n包含的是句子的最后一个单词（也就是最后一个时间步）的隐藏状态，c_n包含的是句子的最后一个单词的细胞状态，所以它们都与句子的长度seq_length无关。
# output[-1]与h_n是相等的，因为output[-1]包含的正是batch_size个句子中每一个句子的最后一个单词的隐藏状态，注意LSTM中的隐藏状态其实就是输出，cell state细胞状态才是LSTM中一直隐藏的，记录着信息

net = torch.nn.LSTM(input_size=20,
                    hidden_size=50,
                    num_layers=5)
output, tuple = net(input_tensor, (hidden_tensor_prev, cell_tensor_prev))
print(net)
print("output")
print(output.size())
print("hn")
print(tuple[0].size())
print("on")
print(tuple[1].size())
# 输出数据包括output,(h_n,c_n):
# output的shape=(seq_length,batch_size,num_directions*hidden_size),
# 它包含的是LSTM的最后一时间步的输出特征(h_t),ｔ是batch_size中每个句子的长度。
# h_n.shape==(num_directions * num_layers,batch,hidden_size)
# c_n.shape==h_n.shape
# h_n包含的是句子的最后一个单词（也就是最后一个时间步）的隐藏状态，c_n包含的是句子的最后一个单词的细胞状态，所以它们都与句子的长度seq_length无关。
# output[-1]与h_n是相等的，因为output[-1]包含的正是batch_size个句子中每一个句子的最后一个单词的隐藏状态，注意LSTM中的隐藏状态其实就是输出，cell state细胞状态才是LSTM中一直隐藏的，记录着信息