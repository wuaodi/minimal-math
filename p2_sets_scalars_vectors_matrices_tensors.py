import torch
print(torch.cuda.is_available())

'------------------------------------------------------------------'
# 在PyTorch中没有直接的集合数据类型，但可以使用Python的集合类型。
# 使用Python内置集合类型
A = {1, 2, 3}
B = {3, 4, 5}

# 集合操作
union = A | B           # 联合
intersection = A & B    # 交集
difference = A - B      # 差集

print("Union:", union)
print("Intersection:", intersection)
print("Difference:", difference)

'------------------------------------------------------------------'
# 在PyTorch中，标量是零维的张量。
scalar = torch.tensor(3.14)
print("Scalar:", scalar)

'------------------------------------------------------------------'
# 向量是一维张量。
vector = torch.tensor([1, 2, 3])
print("Vector:", vector)

'------------------------------------------------------------------'
# 矩阵是二维张量。
matrix = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print("Matrix:\n", matrix)

'------------------------------------------------------------------'
# 张量可以是任意维度的数组。
tensor = torch.tensor([
    [[1, 2], [3, 4]],
    [[5, 6], [7, 8]],
    [[9, 10], [11, 12]]
])
print("Tensor:\n", tensor)

'------------------------------------------------------------------'
# 下面是一些常见的矩阵索引操作：
# 创建一个矩阵
matrix = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print("Matrix:\n", matrix)

# 索引单个元素
element = matrix[1, 2]
print("Element at index (1, 2):", element)

# 索引整行
row = matrix[1, :]
print("Row 1:", row)

# 索引整列
column = matrix[:, 2]
print("Column 2:", column)

# 索引子矩阵
sub_matrix = matrix[0:2, 1:3]
print("Sub-matrix:\n", sub_matrix)

# 修改元素
matrix[0, 0] = 10
print("Modified Matrix:\n", matrix)

# 矩阵的转置
trans = matrix.transpose(0, 1)
print("Transpose:\n", trans)
