import torch
import matplotlib.pyplot as plt
import torchvision
from d2l import torch as d2l
from torch import nn

d2l.set_figsize()
img = d2l.Image.open("../data/RubbishClassification/0/00000.jpg")
d2l.plt.imshow(img)
plt.show()


def apply(img, aug, num_rows=2, num_cols=4, scale=1.5):
    Y = [aug(img) for _ in range(num_rows * num_cols)]
    d2l.show_images(Y, num_rows, num_cols, scale=scale)

# 左右翻转图像通常不会改变对象的类别。这是最早且最⼴泛使⽤的图像增⼴⽅法之⼀。接下来，我们使
# ⽤transforms模块来创建RandomFlipLeftRight实例，这样就各有50%的⼏率使图像向左或向右翻转。
apply(img, torchvision.transforms.RandomHorizontalFlip())
plt.show()

# 下述代码是上下翻转
apply(img, torchvision.transforms.RandomVerticalFlip())
plt.show()

# 随机裁剪
shape_aug = torchvision.transforms.RandomResizedCrop(
    (200, 200),
    scale=(0.1, 1),
    ratio=(0.5, 2))
apply(img, shape_aug)
plt.show()

# 我们还可以创建⼀个RandomColorJitter实例，
# 并设置如何同时随机更改图像的亮度（brightness）、对⽐度（contrast）、饱和度（saturation）和⾊调（hue）。
color_aug = torchvision.transforms.ColorJitter(
    brightness=0.5,
    contrast=0.5,
    saturation=0.5,
    hue=0.5)
apply(img, color_aug)
plt.show()

# 在实践中，我们将结合多种图像增⼴⽅法。
# ⽐如，我们可以通过使⽤⼀个Compose实例来综合上⾯定义的不同的图像增⼴⽅法，
# 并将它们应⽤到每个图像。
augs = torchvision.transforms.Compose([torchvision.transforms.RandomHorizontalFlip(), shape_aug, color_aug])
apply(img=img,
      aug=augs)
plt.show()

