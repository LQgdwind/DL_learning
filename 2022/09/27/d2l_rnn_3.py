# import d2l_rnn_2
import random
import matplotlib.pyplot as plt
import torch
import d2l.torch as d2l

# 假设⻓度为T的⽂本序列中的词元依次为x1, x2, . . . , xT。
# 于是，xt（1 ≤ t ≤ T）可以被认为是⽂本序列在时间步t处的观测或标签。
# 在给定这样的⽂本序列时，
# 语⾔模型（language model）的⽬标是估计序列的联合概率 P(x1, x2, . . . , xT )

# 回想⼀下我们在 8.1节中对⻢尔可夫模型的讨论，
# 并且将其应⽤于语⾔建模
# 如果P(xt+1 | xt, . . . , x1) = P(xt+1 | xt)，
# 则序列上的分布满⾜⼀阶⻢尔可夫性质。阶数越⾼，对应的依赖关系就越⻓。
# 这种性质推导出了许多可以应⽤于序列建模的近似公式：
# P(x1, x2, x3, x4) = P(x1)P(x2)P(x3)P(x4),
# P(x1, x2, x3, x4) = P(x1)P(x2|x1)P(x3|x2)P(x4|x3),
# P(x1, x2, x3, x4) = P(x1)P(x2|x1)P(x3|x1, x2)P(x4|x2, x3).
# 通常，涉及⼀个、两个和三个变量的概率公式分别被称为“⼀元语法”（unigram）、“⼆元语法”（bigram）和“三元语法”（trigram）模型。

tokens = d2l.tokenize(d2l.read_time_machine())
# 因为每个⽂本⾏不⼀定是⼀个句⼦或⼀个段落，因此我们把所有⽂本⾏拼接到⼀起
corpus = [token for line in tokens for token in line]
# tokens 是一个单词行构成的句子列表 ， 每一行的token是一个个字符
vocab = d2l.Vocab(corpus)
print(vocab.token_freqs[:10])

freqs = [freq for token, freq in vocab.token_freqs]
d2l.plot(freqs,
         xlabel='token: x',
         ylabel='frequency: n(x)',
         xscale='log',
         yscale='log')

# 通过此图我们可以发现：词频以⼀种明确的⽅式迅速衰减。将前⼏个单词作为例外消除后，
# 剩余的所有单词⼤致遵循双对数坐标图上的⼀条直线。
# 这意味着单词的频率满⾜⻬普夫定律（Zipf’s law），
# 即第i个最常⽤单词的频率ni为：log ni = −α log i + c

bigram_tokens = [pair for pair in zip(corpus[:-1], corpus[1:])]
bigram_vocab = d2l.Vocab(bigram_tokens)
print(bigram_vocab.token_freqs[:10])

trigram_tokens = [triple for triple in zip(corpus[:-2], corpus[1:-1], corpus[2:])]
trigram_vocab = d2l.Vocab(trigram_tokens)
print(trigram_vocab.token_freqs[:10])

bigram_freqs = [freq for token, freq in bigram_vocab.token_freqs]
trigram_freqs = [freq for token, freq in trigram_vocab.token_freqs]
d2l.plot([freqs, bigram_freqs, trigram_freqs], xlabel='token: x',
         ylabel='frequency: n(x)', xscale='log', yscale='log',
         legend=['unigram', 'bigram', 'trigram'])

# ⾸先，除了⼀元语法词，单词序列似乎也遵循⻬普夫定律，
# 尽管公式中的指数α更⼩（指数的⼤⼩受序列⻓度的影响）。
# 其次，词表中n元组的数量并没有那么⼤，
# 这说明语⾔中存在相当多的结构，这些结构给了我们应⽤模型的希望。
# 第三，很多n元组很少出现，这使得拉普拉斯平滑⾮常不适合语⾔建模。
# 作为代替，我们将使⽤基于深度学习的模型。
plt.show()


# 随机采样
# """使⽤随机抽样⽣成⼀个⼩批量⼦序列"""
def seq_data_iter_random(corpus, batch_size, num_steps):
    # 从随机偏移量开始对序列进⾏分区，随机范围包括num_steps-1
    corpus = corpus[random.randint(0, num_steps - 1):]
    # 减去1，是因为我们需要考虑标签
    num_subseqs = (len(corpus) - 1) // num_steps
    # ⻓度为num_steps的⼦序列的起始索引
    initial_indices = list(range(0, num_subseqs * num_steps, num_steps))
    # 在随机抽样的迭代过程中，
    # 来⾃两个相邻的、随机的、⼩批量中的⼦序列不⼀定在原始序列上相邻
    random.shuffle(initial_indices)

    def data(pos):
    # 返回从pos位置开始的⻓度为num_steps的序列
        return corpus[pos: pos + num_steps]
    num_batches = num_subseqs // batch_size
    for i in range(0, batch_size * num_batches, batch_size):
    # 在这⾥，initial_indices包含⼦序列的随机起始索引
        initial_indices_per_batch = initial_indices[i: i + batch_size]
        X = [data(j) for j in initial_indices_per_batch]
        Y = [data(j + 1) for j in initial_indices_per_batch]
        yield torch.tensor(X), torch.tensor(Y)


my_seq = list(range(35))
for X, Y in seq_data_iter_random(my_seq, batch_size=2, num_steps=5):
    print('X: ', X, '\nY:', Y)

# 顺序分区
# """使⽤顺序分区⽣成⼀个⼩批量⼦序列"""
def seq_data_iter_sequential(corpus, batch_size, num_steps):
    # 从随机偏移量开始划分序列
    offset = random.randint(0, num_steps)
    num_tokens = ((len(corpus) - offset - 1) // batch_size) * batch_size
    Xs = torch.tensor(corpus[offset: offset + num_tokens])
    Ys = torch.tensor(corpus[offset + 1: offset + 1 + num_tokens])
    Xs, Ys = Xs.reshape(batch_size, -1), Ys.reshape(batch_size, -1)
    num_batches = Xs.shape[1] // num_steps
    for i in range(0, num_steps * num_batches, num_steps):
        X = Xs[:, i: i + num_steps]
        Y = Ys[:, i: i + num_steps]
        yield X, Y

for X, Y in seq_data_iter_sequential(my_seq, batch_size=2, num_steps=5):
    print('X: ', X, '\nY:', Y)


# """加载序列数据的迭代器"""
class SeqDataLoader:

    def __init__(self, batch_size, num_steps, use_random_iter, max_tokens):
        if use_random_iter:
            self.data_iter_fn = d2l.seq_data_iter_random

        else:
            self.data_iter_fn = d2l.seq_data_iter_sequential
            self.corpus, self.vocab = d2l.load_corpus_time_machine(max_tokens)
            self.batch_size, self.num_steps = batch_size, num_steps

    def __iter__(self):
        return self.data_iter_fn(self.corpus, self.batch_size, self.num_steps)


# """返回时光机器数据集的迭代器和词表"""
def load_data_time_machine(batch_size, num_steps, use_random_iter=False, max_tokens=10000):
    data_iter = SeqDataLoader(batch_size, num_steps, use_random_iter, max_tokens)
    return data_iter, data_iter.vocab

# 读取⻓序列的主要⽅式是随机采样和顺序分区。
# 在迭代过程中，后者可以保证来⾃两个相邻的⼩批量中的⼦序列在原始序列上也是相邻的。







