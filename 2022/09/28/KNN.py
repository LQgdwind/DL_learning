from numpy import *
import operator

# KNN 主要有两个关键函数
# 一个用于实现KNN:classify_knn
# 一个用于进行数据的归一化：Norm_feature

def Norm_feature(data_set):
    minVal = data_set.min(0)
    maxVal = data_set.max(0)
    ranges = maxVal - minVal  # 计算极差
    # 下一步将初始化一个与原始数据矩阵同尺寸的矩阵
    # 利用tile函数实现扩充向量，并进行元素间的对位运算
    norm_set = zeros(shape(data_set))
    rows = data_set.shape[0]
    norm_set = (data_set - tile(minVal, (rows, 1))) / tile(ranges, (rows, 1))

    return norm_set, ranges, minVal


# 返回极差与最小值留待后续备用
def classify_KNN(test_X, train_set, labels, K):
    rows = train_set.shape[0]
    diff = tile(test_X, (rows, 1)) - train_set
    # tile (a,(m,n))意为把a复制m次为c再把c复制n次后返回
    # 这一行利用tile函数将输入样本实例转化为与训练集同尺寸的矩阵
    # 便之后的矩阵减法运算

    sqDistance = (diff ** 2).sum(axis=1)
    Distance = sqDistance ** 0.5
    sorted_Distance = Distance.argsort()
    # 对每个训练样本与输入的测试样本求欧几里得距离，即点之间的范数
    # 随后按距离由小到大进行排序

    classCount = {}
    for i in range(K):
        vote_label = labels[sorted_Distance[i]]
        classCount[vote_label] = classCount.get(vote_label, 0) + 1
    # dict.get(a,b) 如果a在字典中则返回dict[a]，若a不在字典中则返回b
    # 记录距离最小的前K个类，并存放入列表。KEY对应标签，VALUE对应计数

    sortedClassCount = sorted(classCount.items(),
                              key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]