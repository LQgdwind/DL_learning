import d2l.torch as d2l
import torch

x = torch.tensor([[2., 1.],
                  [3., 2.]], requires_grad=True)
y = torch.tensor([[3., 2.],
                  [4., 4.]], requires_grad=True)
z = x ** 2 + y ** 3
z.backward(torch.tensor([[1, 2],
                         [1, 1]]))
print(x.grad)

