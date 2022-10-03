import numpy as np

# 二维matrix

# 对于二维matrix，我们可以使用
# dtype观察他的数据类型
# shape观察他的形状

# 初始化
a = np.array([[1, 2, 3], [4, 5, 6]])
b = np.zeros((3, 2))
c = np.ones((3, 2))
d = np.eye((3))
# eye方法生成dtype=np.float64的单位矩阵
e = np.full((3, 2), 100)
f = np.empty((3, 2))

# 随机初始化方法与一维的类似
# 有rand/randint/uniform/randn/normal

# 在matrix中,
# axis=0代表列,
# axis=1代表行,
# 默认axis=0
g = np.arange(6).reshape((2, 3)).astype(np.float64)
print(g)
print(g.sum())
print(g.sum(axis=0))
print(g.sum(axis=1))

# 基本运算
# +、-、*、/、**、//、@
# 注意 *是按元素乘法、@是矩阵乘法

# 转置操作
g = g.T
print(g)

# reshape 改变形态
# 如果某一个轴为-1，代表自适应
print(g.reshape(-1, 1))
print(g.reshape(1, -1))

# 矩阵合并
# hstack横向，vstack纵向，也可以理解为堆叠
h = np.linspace(1,12,12).reshape((3,4)).astype(np.float64)
i = np.linspace(1,6,6).reshape((3,2)).astype(np.float64)
print(np.hstack((h,i)))

# 矩阵分裂hsplit和vsplit
i,j = np.hsplit(h,[2])
print(i)
print(j)

# matrix的复制操作，
# tile整个复制，repeat可以理解为挨个复制
k = np.arange(4).reshape((2,2))
print(np.tile(k,(3,4)))
print(k.repeat(4,axis=1))


# delete删除操作
l = np.arange(15).reshape((3,5))
print(np.delete(l,[1,3],axis=1))
print(np.delete(l,1,axis=0))

# insert插入操作
m = np.arange(9).reshape((3,3))
n = np.arange(6).reshape((3,2))
print(np.insert(m,[0,2],n,axis=1))

# append操作，只能在末尾操作
print(np.append(m,np.zeros((3,2)),axis=1))
