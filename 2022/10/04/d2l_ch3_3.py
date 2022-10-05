import torch
import torchvision
from torch.utils import data
from torchvision import transforms
from d2l import torch as d2l

trans = transforms.ToTensor()
mnist_train = torchvision.datasets.FashionMNIST(
    root="../../09/data",
    train=True,
    transform=trans,
    download=False)
mnist_test = torchvision.datasets.FashionMNIST(
    root="../../09/data",
    train=False,
    transform=trans,
    download=False
)

print("train_set: {arg1}, test_set: {arg2}".format(arg1=len(mnist_train),arg2=len(mnist_test)))
