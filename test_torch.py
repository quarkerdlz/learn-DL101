import torch
import numpy as np
# from matplotlib_inline import backed_inline
from matplotlib import pyplot as plt
import d2lzh_pytorch as d2l
# print(torch.__version__)

'''创建tensor'''


'''60分钟入门pytorch'''


# 创建tensor
# x = torch.arange(12) # 创建⼀个⾏向量x
# x.shape # torch.Size([12])
# x.view(-1, 1) # 相当于增维，只要指定一个参数如高或宽，就能自动计算另一个；同时可以用reshape
# x.numel() # 检查元素总数，必须和size相同
# x.sum()对所有函数求和

# 创建全0或全1的tensor
# torch.zeros(2,3) 和 torch.zeros((2,3))没有区别
# torch.zeros(2, 3, 4) # 后面两维就是矩阵的shape,以此类推
# torch.ones((2, 3, 4))

# 创建常数值矩阵
# shape=(2, 3,) # ,在只有一个元素时，把它转换为元组
# rand_tensor = torch.rand(shape)
# randn是均值为0，方差为1的正态分布；rand是[0,1)均匀分布
# one_tensor = torch.ones(shape)
# zeros_tensor = torch.zeros(shape)

# print(rand_tensor)
# print(one_tensor)
# print(zeros_tensor)

# 创建一个5x3未初始化的tensor
# x = torch.empty(5, 3)
# print(x)

# 创建一个5x3的随机初始化的tensor
# x = torch.rand(5, 3)
# print(x)

# 创建一个5x3的long型全0的tensor
# x = torch.zeros(5, 3, dtype=torch.long)
# print(x)

# 根据数据创建：
# x = torch.tensor([5.5, 3])
# print(x)

# 运算符（+、-、*、/和**）都按元素运算
# x = torch.tensor([1.0, 2, 4, 8])
# y = torch.tensor([2, 2, 2, 2])
# x + y # x和y的维数要相同


# 查看tensor的shape、datatype和device
# tensor = torch.rand(3, 4)
# print(tensor.shape)
# print(tensor.dtype)
# print(tensor.device) # cpu
# print(x_data.dtype)

# 将多个张量连接在一起
# X = torch.arange(12, dtype=torch.float32).reshape((3,4))
# x = torch.randn_like(x, dtype=torch.float) # 指定新的数据类型
# print(x)
# Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
# torch.cat((X, Y), dim=0), torch.cat((X, Y), dim=1)

# 通过逻辑运算符构建⼆元张量，满足条件为1，不满足为0
# X == Y 
# a = (X == Y).float() # 将值转换为1和0

# 广播机制
# x = torch.arange(1,3).view(1,2) # tensor([[1, 2]]) 自动增加一个维度,便于广播机制
# print(torch.arange(1,3)) # tensor([1, 2])
# y = torch.arange(1, 4).view(3, 1) # tensor([[1],[2],[3]])
# print(torch.arange(1, 4)) # tensor([1, 2, 3])
# print(x + y) # tensor([[2, 3],[3, 4],[4, 5]])
# # x的第一行两个元素复制到2、3行，y的第一列3个元素复制到第二列
# 本机制适用于一维tensor之间或者tensor和标量之间？

# 索引和切⽚
# X[-1], X[1:3]
# X[1, 2] = 9 # 指定索引来将元素写⼊矩阵
# 批量写入
# X[0:2, :] = 12

# 带_为原地操作符，不用复制，直接在原来内存上改变值
# print(tensor, "\n")
# tensor.add_(5)
# print(tensor)

# Y = Y + X # 结果Y会重新分配存储地址
# before = id(Y) # id表示存储位置
# Y = Y + X
# id(Y) == before # 如果不原地更新就会无意中引用旧的参数
# 所以使用原地操作符_，不用复制，直接在原来内存上改变值
# print(tensor, "\n")
# tensor.add_(5)
# print(tensor)
# Z = torch.zeros_like(Y)
# print('id(Z):', id(Z))
# Z[:] = X + Y 
# print('id(Z):', id(Z))
# Z.add_(X + Y)
# print('id(Z):', id(Z))
# before = id(X)
# X += Y # 也相当于原地操作
# id(X) == before

# tensor可以从np中取得，或者其它tensor
# data = [[1,2], [3,4]]
# x_data = torch.tensor(data)
# print(x_data)
# cpu中的tensor可以和np互相分享内存地址，改变一个另一个也会变
# t = torch.ones(5)
# print(t)
# n = t.numpy()
# print(n)
# t.add_(1)
# print(t)
# print(n) # n也会跟着t变

# array
# n = np.ones(5)
# t = torch.from_numpy(n)
# np.add(n, 1, out=n) # out表示将结果存储到什么位置
# print(t)
# print(n)

# A = X.numpy()
# B = torch.tensor(A)
# type(A), type(B)

# a = torch.tensor([3.5])
# a, a.item(), float(a), int(a)

# np_array = np.array(data)
# x_np = torch.from_numpy(np_array)
# tensor2array = torch_data.numpy() # torch到numpy
# print(x_np)

# 从其它tensor来
# x_ones = torch.ones_like(x_data)
# print(x_ones)

# x_rand = torch.rand_like(x_data, dtype=torch.float)
# # 要注明type，不然int是不能random的
# print(x_rand)

# 根据现有tensor创建，会默认一些原tensor属性，如数据类型
# x = torch.zeros(5, 3, dtype=torch.float)
# x = x.new_ones(5, 3, dtype=torch.float64) # 
# # 返回的tensor默认具有相同的torch.dtype和torch.device
# print(x)
# print(x.size())
# print(x.shape)

# a = torch.Tensor(*[5, 3])
# print(a)
# b = torch.eye(*[5, 5]) # 对角线矩阵
# print(b)

# 数据预处理
import os

# os.makedirs(os.path.join('../ML&DL', 'data'), exist_ok=True)
# # os.path.join初拼接路径
# # os.makedirs创建path下的目录（文件夹）
# data_file = os.path.join('../ML&DL', 'data', 'house_tiny.csv')
# with open(data_file, 'w') as f:
# 	f.write('NumRooms,Alley,Price\n') # 列名
# 	f.write('NA,Pave,127500\n') # 每⾏表⽰⼀个数据样本
# 	f.write('2,NA,106000\n')
# 	f.write('4,NA,178100\n')
# 	f.write('NA,NA,140000\n')

# import pandas as pd 

# data = pd.read_csv(data_file)
# # print(data)

# '''处理缺失值'''
# inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]
# inputs = inputs.fillna(inputs.mean())
# # print(inputs)

# # 对于类别值，这里只有pave和nan，所以对所有状态分别设置为0和1, 是一种onehot编码
# inputs = pd.get_dummies(inputs, dummy_na=True)
# # print(inputs)

# # 转换为张量格式
# X, y = torch.tensor(inputs.values), torch.tensor(outputs.values)
# print(X, y)

# tensor的操作
# 用方法to()可以将Tesnor在CPU和GPU之间相互移动
# if torch.cuda.is_available():
# 	tensor = tensor.to('cuda')
# 	print(tensor.device)

'''用方法to()可以将Tesnor在CPU和GPU之间相互移动'''
# 以下代码只有在pytorch gpu版本才能执行
# x = torch.zeros(5, 3, dtype=torch.long)
# if torch.cuda.is_available():
# 	device = torch.device("cuda") # GPU
# 	y = torch.ones_like(x, device=device)
# 	print(y.device)
# 	# 直接创建一个在GPU上的Tensor
# 	x = x.to(device) # 等价于.to("cuda")
# 	z = x + y
# 	print(z)
# 	a = z.to("cpu", torch.double)
# 	# pinrt(z.to("cpu", torch.double))
# 	print(a.device) # to()还可以同时更改数据类型


# 跟numpy一样的切片
# tensor = torch.ones(4, 4)
# tensor[:, 1] = 2
# print(tensor)

# 合并矩阵, dim范围是[-2,1]，0=-2, 1=-1 
# t0 = torch.cat([tensor, tensor, tensor], dim=0)
# t1 = torch.cat([tensor, tensor, tensor], dim=1)
# t2 = torch.cat([tensor, tensor, tensor], dim=-2)
# print(t0)
# print(t1)
# print(t2)
# print(torch.cat([tensor, tensor, tensor], dim=-1))


'''线性代数'''
# 按元素运算
A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
B = A.clone() # 通过分配新内存，将A的⼀个副本分配给B
# print(A + B)
# # 按元素相乘，Hadamard积
# print(A * B)
# # 和标量相乘
# a = 2
# X = torch.arange(24).reshape(2, 3, 4)
# print(a+X)
# print((a*X).shape)

# 指定某一维求和以降维
# A_sum_axis0 = A.sum(axis=0)
# print(A_sum_axis0)
# print(A_sum_axis0.shape)
# A_sum_axis1 = A.sum(axis=1)
# print(A_sum_axis1)
# print(A_sum_axis1.shape)

# # A.sum(axis=[0, 1]) # 等于A.sum()
# # 同样的情况可类比于A.mean()
# print(A.mean(axis=0))
# print(A.mean(axis=1))
# 非降维求和
# sum_A = A.sum(axis=1, keepdims=True)
# print(sum_A.shape)
# print(A / sum_A) # 用了广播机制

# 按某一维度求和
# print(A.cumsum(axis=0)) # 按行求和，第一行加到第二行，以此类推
# print(A.cumsum(axis=1))

# 向量点积，即相同位置相乘后求和
x = torch.tensor([1.0, 2, 4, 8])
y = torch.ones(4, dtype=torch.float32)
# print(torch.dot(x, y))

# 矩阵-向量积
# print(torch.mv(A, x))
# print(torch.mv(x, A)) # 报错

# 矩阵乘法
# print(tensor.mul(tensor)) # 每个对应元素相乘
# print(tensor * tensor) # 与上面的相同
B = torch.ones(4, 3)
# print(torch.mm(A, B))

# 正常的矩阵乘法
# print(tensor.matmul(tensor.T))
# print(tensor@tensor.T)

# 范数
# 一般为L2范数
u = torch.tensor([3.0, 4.0])
print(torch.norm(u))

# L1范数
print(torch.abs(u).sum())

# Frobenius范数
print(torch.norm(torch.ones(4,9)))






















