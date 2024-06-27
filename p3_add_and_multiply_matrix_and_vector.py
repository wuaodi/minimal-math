import torch

# 用pytorch实现两个向量的相加
a = torch.tensor([1, 3])
b = torch.tensor([4, 2])
c = a + b
print(f"两个向量相加结果: {c}")

# 用pytorch实现两个矩阵相加
a = torch.tensor([[1, 3], [2, 4]])
b = torch.tensor([[4, 2], [3, 1]])
c = a + b
print(f"两个矩阵相加结果: \n{c}")

# 用pytorch实现矩阵和标量的相乘和相加
a = torch.tensor([[1, 3], [2, 4]])
b = 2
c = 3
d = a * b + c
print(f"矩阵与标量相乘后加{c}的结果: \n{d}")

# 用pytorch实现矩阵和向量的相加，使用广播机制
a = torch.tensor([[1, 3], [2, 4]])
b = torch.tensor([4, 2])
c = a + b
print(f"矩阵与向量广播相加结果: \n{c}")

# 用pytorch实现两个矩阵的相乘
a = torch.tensor([[3, 1], [2, 1]])
b = torch.tensor([[1, 2], [3, 1]])
c = torch.matmul(a, b)
print(f"两个矩阵相乘结果: \n{c}")

# 用pytorch实现两个矩阵的点乘
a = torch.tensor([[3, 1], [2, 1]])
b = torch.tensor([[1, 2], [3, 1]])
c = a * b
print(f"矩阵元素逐点相乘(方式一)结果: \n{c}")
c = torch.mul(a, b)
print(f"矩阵元素逐点相乘(方式二)结果: \n{c}")

# 用pytorch实现两个向量的内积
a = torch.tensor([1, 3])
b = torch.tensor([4, 2])
c = torch.dot(a, b)
print(f"两个向量的内积结果: {c}")

# 用pytorch实现两个向量outer product
a = torch.tensor([1, 3])
b = torch.tensor([4, 2])
c = torch.outer(a, b)
print(f"两个向量的外积(Outer Product)结果: \n{c}")

# 用pytorch演示矩阵的加法分配律
a = torch.tensor([[1, 3], [2, 4]])
b = torch.tensor([[4, 2], [3, 1]])
c = torch.matmul(a, b) + torch.matmul(a, b)
print(f"矩阵乘法后加自身结果: \n{c}")
c = torch.matmul(a, (b + b))
print(f"先加后乘结果(验证分配律): \n{c}")

# 用pytorch演示矩阵的结合律
a = torch.tensor([[1, 3], [2, 4]])
b = torch.tensor([[4, 2], [3, 1]])
c = torch.matmul(a, torch.matmul(b, a))
print(f"结合律验证(顺序ABC)结果: \n{c}")
c = torch.matmul(torch.matmul(a, b), a)
print(f"结合律验证(顺序ACB)结果: \n{c}")

# 矩阵不满足交换律，因为维度的约束

# 用pytorch演示两个矩阵相乘的转置等于交换位置转置后相乘
a = torch.tensor([[1, 3], [2, 4]])
b = torch.tensor([[4, 2], [3, 1]])
c = torch.matmul(a, b).transpose(0, 1)
print(f"A*B转置结果: \n{c}")
c = torch.matmul(b.t(), a.t())
print(f"B转置*A转置结果: \n{c}")

# 用pytorch实现线性方程组求解
A = torch.tensor([[2., 1.], [1., -3.]])
b = torch.tensor([7., -9.])
x = torch.linalg.solve(A, b)
print(f"线性方程组Ax=b的解为: {x}")

# 通义灵码确实好用，让编程更加高效便捷！
