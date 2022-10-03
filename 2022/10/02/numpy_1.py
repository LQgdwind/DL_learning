import numpy as np

a = np.array([1., 2., 3.])
b = np.zeros(3, dtype=np.int32)
c = np.zeros_like(a)

# 所有创建包含固定值vector的方法都有_like函数
# 比如ones_like,zeros_like,full_like,empty_like

# np.linspace(start=, stop=, num=, endpoint=)
# 创建等差数列
d = np.linspace(start=1,
                stop=10,
                num=10,
                endpoint=True)
print(d)

# np.arange(start=, stop=, step= )
# 创建整数等差数列
e = np.arange(start=1,
              stop=10,
              step=2)
print(e)

# 若需要用arange方法创建浮点型的，可以使用
# np.arange(args).astype(np.float32)
f = np.arange(3).astype(np.float32)
print(f)

# 生成随机array
g = np.random.randint(low=0,
                      high=10,
                      size=7)
# random.randint方法生成size个[low,high)的随机整数array
# 注意是前闭后开

h = np.random.rand(3)
# random.rand方法生成n个[0,1)的随机浮点数array

i = np.random.uniform(low=1,
                      high=10,
                      size=3)
# random.uniform生成size个[low,high)的随机浮点数array

j = np.random.randn(5)
# random.randn()方法生成n个符合μ=0,σ=1的正态分布array

k = np.random.normal(loc=5,
                     scale=2,
                     size=3)
# random.normal方法生成size个μ=loc,σ=scale的正态分布array

# 向量索引操作只展示部分数据，不改变数据本身
l = np.arange(1, 6).astype(np.float32)
print(l[4:2:-1])

# 向量布尔操作
print(np.any(l>=5))
print(np.all(l>=5))
print(l>=5)
print(l[l>=5])
print((l>=1)&(l<=2))
print(l[(l>=1) & (l<=2)])

# 可以用where方法来索引或替换，用clip方法来指定上下界
print(np.where(l>4))
print(np.clip(l,
              a_min=3,
              a_max=4))
# np.clip操作返回一个新的array
print(l)

# numpy的优势就是把vector当做数做整体运算，避免循环运算
# 常用的有+、-、*、/、**、//、
# dot、log、exp、cross(外积)、sqrt、
# sin、cos、tan、
# arcsin、arccos、arctan、hypot(给两个列表，计算对应位置的勾股定理斜边)、
# floor(下取整)、ceil(上取整)、round(四舍五入,.5也是舍)

# numpy还可以做基础的统计操作，
# 比如max，min，mean，sum，var，std等

# numpy提供查找操作
# 其中有三种方法：
# where，难懂且对于x处于array末端很不友好
# next，相对较快，但需要numba
# searchsorted，针对于已排过序的array

