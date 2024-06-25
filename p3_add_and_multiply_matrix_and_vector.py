import torch
import numpy as np

# 用pytorch实现两个向量的相加
a = torch.tensor([1, 3])
b = torch.tensor([4, 2])
c = a + b
print(c)

# 用pytorch实现两个矩阵相加
a = torch.tensor([[1, 3], [2, 4]])
b = torch.tensor([[4, 2], [3, 1]])
c = a + b
print(c)

# 用pytorch实现矩阵和标量的相乘和相加
a = torch.tensor([[1, 3], [2, 4]])
b = 2
c = 3
d = a * b + c
print(d)

# 用pytorch实现矩阵和向量的相加，使用广播机制
a = torch.tensor([[1, 3], [2, 4]])
b = torch.tensor([4, 2])
c = a + b
print(c)

# 用pytorch实现两个矩阵的相乘
a = torch.tensor([[3, 1],
                  [2, 1]])
b = torch.tensor([[1, 2],
                  [3, 1]])
c = torch.matmul(a, b)
print(c)

# 用pytorch实现两个矩阵的点乘
a = torch.tensor([[3, 1],
                  [2, 1]])
b = torch.tensor([[1, 2],
                  [3, 1]])
c = a * b
print(c)
c = torch.mul(a, b)
print(c)

# 用pytorch实现两个向量的内积
a = torch.tensor([1, 3])
b = torch.tensor([4, 2])
c = torch.dot(a, b)
print(c)

# 用pytorch实现两个向量outer product
# a列向量 ✖ b行向量 = c矩阵
a = torch.tensor([1, 3])
b = torch.tensor([4, 2])
c = torch.outer(a, b)
print(c)

# 用pytorch演示矩阵的加法分配律
a = torch.tensor([[1, 3],
                  [2, 4]])
b = torch.tensor([[4, 2],
                  [3, 1]])
c = torch.matmul(a, b) + torch.matmul(a, b)
print(c)
c = torch.matmul(a, (b + b))
print(c)

# 用pytorch演示矩阵的结合律
a = torch.tensor([[1, 3],
                  [2, 4]])
b = torch.tensor([[4, 2],
                  [3, 1]])
c = torch.matmul(a, torch.matmul(b, a))
print(c)
c = torch.matmul(torch.matmul(a, b), a)
print(c)

# 矩阵不满足交换律，因为维度的约束

# 用pytorch演示两个矩阵相乘的转置 等于 交换位置转置后相乘
a = torch.tensor([[1, 3],
                  [2, 4]])
b = torch.tensor([[4, 2],
                  [3, 1]])
c = torch.matmul(a, b).transpose(0, 1)
print(c)
c = torch.matmul(b.t(), a.t())
print(c)

# 用pytorch实现线性方程组求解
# 定义系数矩阵A和常数项向量b
A = torch.tensor([[2., 1.],
                  [1., -3.]])
b = torch.tensor([7., -9.])
# 使用torch.linalg.solve求解线性方程组 Ax = b
x = torch.linalg.solve(A, b)
print("解为：", x)

# 通义灵码还可以，好用！
