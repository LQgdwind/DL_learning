import torch
# Dataset是一个抽象类，为了能够方便的读取，
# 需要将要使用的数据包装为Dataset类。
# 自定义的Dataset需要继承它并且实现两个成员方法：
# 1. __getitem__() 该方法定义用索引(0 到 len(self))获取一条数据或一个样本
# 2. __len__() 该方法返回数据集的总长度

from torch.utils.data import Dataset
import pandas as pd

class BulldozerDataset(Dataset):
    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file)
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        return self.df.iloc[idx].SalePrice

ds_demo = BulldozerDataset('../data/bluebook-for-bulldozers/median_benchmark.csv')
# 相对路径
print(len(ds_demo))
print(ds_demo[0])


# DataLoader为我们提供了对Dataset的读取操作，
# 常用参数有：batch_size(每个batch的大小)、
# shuffle(是否进行shuffle操作)、
# num_workers(加载数据的时候使用几个子进程)。
# 下面做一个简单的操作

dl = torch.utils.data.DataLoader(ds_demo,
                                 batch_size=10,
                                 shuffle=True,
                                 num_workers=0)

# DataLoader返回的是一个可迭代对象，我们可以使用迭代器分次获取数据
idata = iter(dl)
print(next(idata))

# 常见的用法是使用for循环对其进行遍历
for i, data in enumerate(dl):
    print(i, data)
    # 为了节约空间，只循环一遍
    break


# torchvision是PyTorch中专门用来处理图像的库
# torchvision.datasets 可以理解为PyTorch团队自定义的dataset，
# 这些dataset帮我们提前处理好了很多的图片数据集，
# 我们拿来就可以直接使用： - MNIST - COCO - Captions - Detection - LSUN - ImageFolder - Imagenet-12 - CIFAR - STL10 - SVHN - PhotoTour 我们可以直接使用，示例如下：

import torchvision.datasets as datasets
trainset = datasets.MNIST(root='../data',
                          train=True,
                          download=True,
                          transform=None)

# torchvision不仅提供了常用图片数据集，
# 还提供了训练好的模型，
# 可以加载之后，直接使用，
# 或者在进行迁移学习 torchvision.models模块的子模块中包含以下模型结构。
# - AlexNet - VGG - ResNet - SqueezeNet - DenseNet

#我们直接可以使用训练好的模型，当然这个与datasets相同，都是需要从服务器下载的
import torchvision.models as models
resnet18 = models.resnet18(pretrained=True)


# transforms 模块提供了一般的图像转换操作类，用作数据处理和数据增强

from torchvision import transforms as transforms
transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),  #先四周填充0，在把图像随机裁剪成32*32
    transforms.RandomHorizontalFlip(),  #图像一半的概率翻转，一半的概率不翻转
    transforms.RandomRotation((-45,45)), #随机旋转
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.229, 0.224, 0.225)), #R,G,B每层的归一化用到的均值和方差
])

