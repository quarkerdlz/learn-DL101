'''本节主要是学习autograd'''

import torch, torchvision
import numpy as np
# from matplotlib_inline import backend_inline
from matplotlib import pyplot as plt
import d2lzh_pytorch as d2l
# from torch.autograd import Variable # torch中variable模块

# model = torchvision.models.resnet18(pretrained=False) # false只加载结构，不用在线下载
# pre = torch.load("E:\\DFIM\\McIntosh\\ML&DL\\resnet18-f37072fd.pth")
# model.load_state_dict(pre)

# # 创建随机数据tensor代表单个图像，设置长宽、通道
# # 通道为1就是灰度，通道为3是RGB
# data = torch.rand(1, 3, 64, 64)
# # print(data.shape)
# # 设置图像对应的初始标签值
# labels = torch.rand(1, 1000)
# # print(labels.shape)

# # 把input输入进model的第一隐藏层，得到预测值
# prediction = model(data)
# # print(prediction.shape) #与label的shape一样

# # 计算error，然后反向传递误差
# # autograd计算和存储每个模型参数的梯度
# loss = (prediction - labels).sum()
# loss.backward()

# # 加载优化器，这里使用SGD，将所有参数放入优化器
# optim = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)

# # 用.step()发起梯度，优化器用梯度调整好每一个参数
# optim.step()

a = torch.tensor([2., 3.], requires_grad=True) # 加.变为float
b = torch.tensor([6., 4.], requires_grad=True)
'''
requires_grad表示变量要不要送到back propaganda中，
设置为true时，将追踪在tensor上所有操作，这样就能计算并传播梯度
设置为False，将来的计算就不会被track，梯度就传不过去
'''

# 假设Q是loss function
# Q = 3*a**3 - b**2

# '''
# y.backward()，如果y是标量，则不需要为backward()传入任何参数；
# 否则需要输入一个与y同shape的tensor
# '''
# external_grad = torch.tensor([1., 1.])
# Q.backward(gradient=external_grad) # 可以不加gradient

# print(9*a**2 == a.grad)
# print(-2*b == b.grad)
 
'''标量变量的反向传播'''
x = torch.arange(4.0)
x.requires_grad_(True) # 等价于x=torch.arange(4.0,requires_grad=True)
x.grad # 默认值是None

# y = 2*torch.dot(x, x)
# # print(y)
# y.backward()
# # print(x.grad)
# # print(x.grad==4*x)

# x.grad.zero_()
# y = x.sum()
# y.backward()
# print(x.grad) # 如果不清零就是tensor([ 1.,  5.,  9., 13.])

'''非标量变量的反向传播'''
# x.grad.zero_()
# y = x*x # 按元素相乘，Hadamard积
# y.sum().backward() # 等价于y.backward(torch.ones(len(x)))
# y.mean().backward() # 多了一步的除法，除以个数
# print(x.grad)

# '''
# 有向无环图，由function类和tensor组成，记录整个计算过程
# 每个tensor都有.grad_fn属性，即创建该tensor的function
# 即该tensor是否是某种运算得到，若是则grad_fn返回一个与这些运算相关的对象；否则为None
# '''
# # x = torch.ones(2, 2, requires_grad=True)
# print(x)
# print(x.grad_fn) # 返回None, x为叶节点

# y = x + 2
# # print(y)
# # print(y.grad_fn) 

# z = y * y * 3
# out = z.mean()
# # print(z, out)


# a = torch.randn(2, 2)
# a = ((a*3) / (a-1))
# print(a.requires_grad)
# a.requires_grad_(True) # 原地改变requires_grad属性
# print(a.requires_grad)
# b = (a*a).sum()
# print(b.grad_fn)

# x = torch.rand(5, 5)
# y = torch.rand(5, 5)
# z = torch.rand((5, 5), requires_grad=True)

# a = x + y
# print(a.requires_grad)
# b = x + z
# print(b.requires_grad) # 只要有一个参数为true，整个为true

# x = torch.ones(2, 2, requires_grad=True)
# y = x + 2
# z = y * y * 3
# out = z.mean()

# out.backward() 
# # print(x.grad) # out关于x的梯度

# # 梯度在反向传播时是累加的，所以一般反向传播前要把梯度清零
# out2 = x.sum()
# out2.backward()
# print(x.grad) # 此时的结果是加到之前的结果上

# out3 = x.sum()
# x.grad.data.zero_()
# out3.backward()
# print(x.grad)

'''
不允许张量对张量求导，只允许标量对张量求导
求导结果和自变量必须是同样的shape
所以要把张量通过所有张量元素加权求和的方式转换为标量
'''
# x = torch.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
# y = 2 * x
# z = y.view(2, 2)
# # print(z)

# # y不是标量所以调用backward时要传入一个和y同形的权重向量进行加权求和得到一个标量
# v = torch.tensor([[1.0, 0.1],[0.01, 0.001]], dtype=torch.float)
# # v = torch.tensor([1.0, 0.1, 0.01, 0.001], dtype=torch.float)
# z.backward(v)
# print(x.grad) # 这里x.gard和x同shape

'''分离计算'''
# x.grad.zero_()
y = x*x
u = y.detach() # 分离y来返回⼀个新变量u,同时z将u作为常数处理
z = u*x

z.sum().backward()
# print(x.grad == u)

x.grad.zero_()
y.sum().backward()
# print(x.grad == 2 * x) # 说明记录了y的计算结果，可以在y上调用反向传播

# 中断梯度追踪
# x = torch.tensor(1.0, requires_grad=True)
# y1 = x**2
# with torch.no_grad():
# 	y2 = x**3
# y3 = y1 + y2

# # print(x.requires_grad)
# # print(y1, y1.requires_grad)
# # print(y2, y2.requires_grad)
# # print(y3, y3.requires_grad)

# y3.backward()
# print(x.grad) # 输出是2，因为y2没有回传，所以被排除在DAG之外

# 如果要修改tensor，但不想被autograd记录
# x = torch.tensor(1.0, requires_grad=True)
# print(x.data)
# print(x.data.requires_grad)

# y = 2 * x
# x.data *= 100 # 只改变了值，不记录在DAG中

# y.backward()
# print(x) # 更改data会影响tensor的值
# print(x.grad)

'''导数和微分'''
# def f(x):
# 	return 3*x **2 - 4*x

# def numerical_lim(f, x, h):
# 	return (f(x+h)-f(x)) / h

# h = 0.1
# for i in range(5):
# 	print(f'h={h:.5f}, numerical limit={numerical_lim(f, 1, h):.5f}')
# 	h*=0.1



# X = np.arange(0, 3, 0.1)
# X, Y = [[]] * len(X), X
# for x, y in zip(X, Y):
# 	if len(x):
# 		plt.plot(x, y)
# 	else:
# 		plt.plot(y)
# plt.show()

'''画图'''
# x = np.arange(0, 3, 0.1)
# d2l.plot(x, [f(x), 2 * x - 3], 'x', 'f(x)', legend=['f(x)', 'Tangent line (x=1)'])

'''Python控制流的梯度计算'''
# def f(a):
# 	b = a*2
# 	while b.norm() < 1000:
# 		b = b*2
# 	if b.sum() > 0:
# 		c = b
# 	else:
# 		c = 100*b
# 	return c

# a = torch.randn(4, requires_grad=True)
# d = f(a)
# # v = torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=torch.float)
# d.sum().backward()

# f是一个分段线性函数，对于任何a，存在某个常量标量k，使得f(a)=k*a，其中k的值取决于输⼊a
# print(a.grad == d / a)


'''概率'''
# 为了抽取⼀个样本，即掷骰⼦，只需传⼊⼀个概率向量。
# 输出是另⼀个相同⻓度的向量：它在索引i处的值是采样结果中i出现的次数。
from torch.distributions import multinomial
fair_probs = torch.ones([6]) / 6
# print(multinomial.Multinomial(1, fair_probs).sample())
# 第一个参数是试验次数，第二个是试验概率
# 增加试验次数
# print(multinomial.Multinomial(10, fair_probs).sample())

# 模拟1000次
# counts = multinomial.Multinomial(1000, fair_probs).sample()
# print(counts/1000)

counts = multinomial.Multinomial(10, fair_probs).sample((500,))
# print(counts)
cum_counts = counts.cumsum(dim=0) # 把500组试验的每种情况累加
estimates = cum_counts / cum_counts.sum(dim=1, keepdims=True) # 把500组情况，各次的情况总数累加
# 并算出每种情况的频率，此矩阵就是每种情况频率随着试验增多的变化
# print(estimates)

d2l.set_figsize((6, 4.5))
for i in range(6):
	plt.plot(estimates[:,i].numpy(), label=("P(die=" + str(i + 1) + ")"))
plt.axhline(y=0.167, color='black', linestyle='dashed')
plt.gca().set_xlabel('Groups of experiments')
plt.gca().set_ylabel('Estimated probability')
d2l.plt.legend()
plt.show()









'''numpy和torch之间的相互转换'''
# np_data = np.arange(6).reshape((2, 3))
# torch_data = torch.from_numpy(np_data) # numpy转换成tensor
# tensor2array = torch_data.numpy() # tensor转换成np.array
# print(
# 	'\nnumpy array:', np_data,
# 	'\ntorch tensor:', torch_data,
# 	'\ntensor to array:', tensor2array,
# 	)

# 绝对值abs
# data = [-1, -2, 1, 2]
# tensor = torch.FloatTensor(data)
# print(
#     '\nabs',
#     '\nnumpy: ', np.abs(data),          # [1 2 1 2]
#     '\ntorch: ', torch.abs(tensor)      # [1 2 1 2]
# )

# 其它函数
# print(
#     '\nabs',
#     '\nnumpy: ', np.sin(data),         
#     '\ntorch: ', torch.sin(tensor)      
# )

# print(
#     '\nmean',
#     '\nnumpy: ', np.mean(data),         # 0.0
#     '\ntorch: ', torch.mean(tensor)     # 0.0
# )

'''矩阵运算'''
# data = [[1,2], [3,4]]
# tensor = torch.FloatTensor(data)
# data = np.array(data)
# print(
# 	'\nmatrix multiplication (matmul)',
# 	'\nnumpy: ', np.matmul(data, data),
# 	'\ntorch:', torch.mm(tensor, tensor),
# 	'\nnumpy: ', data.dot(data),
# 	'\ntorch:', tensor.reshape(1,4)[0].dot(tensor.reshape(1, 4)[0])
# )
# 最后一个相当于展平，每一个元素和对应的相乘然后求和
# tensor.dot(tensor) 只能针对一维数组

'''在 Torch 中的 Variable 就是一个存放会变化的值的地理位置.
里面的值会不停的变化. 就像一个裝鸡蛋的篮子, 鸡蛋数会不停变动.
那谁是里面的鸡蛋呢, 自然就是Torch的Tensor. '''
 
# 如果用一个Variable进行计算, 那返回的也是一个同类型的Variable.

# 先生鸡蛋
# tensor = torch.FloatTensor([[1,2],[3,4]])
# # 把鸡蛋放到篮子里，
# variable = Variable(tensor, requires_grad=True)
# # 误差反向传播用variable搭建的计算图进行传播
# # requires_grad表示变量要不要送到back propaganda中
# # tensor.requires_grad_(requires_grad = True)
# # tensor.requires_grad为初始的梯度

# print(tensor)
# print(variable)

# t_out = torch.mean()



















