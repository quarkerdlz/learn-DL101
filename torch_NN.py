'''这一节学习Neural networks的搭建'''

import torch
import torch.nn as nn
import torch.nn.functional as F 


'''用pytorch做线性回归'''
from matplotlib import pyplot as plt
import numpy as np
import random

'''生成数据集'''
# num_inputs = 2
# num_examples = 1000
# true_w = [2, -3.4]
# true_b = 4.2
# features = torch.from_numpy(np.random.normal(loc=0, scale=1, size=[num_examples, num_inputs]))
# labels = true_w[0]*features[:, 0] + true_w[1]*features[:, 1] + true_b
# labels += torch.from_numpy(np.random.normal(loc=0, scale=0.01, size=labels.size()))



# plt.scatter(features[:, 1].numpy(), labels.numpy(), 1);
# plt.show()

'''读取数据'''
# def data_iter(batch_size, features, labels):
# 	num_examples = len(features)
# 	indices = list(range(num_examples))
# 	random.shuffle(indices)
# 	for i in range(0, num_examples, batch_size):
# 		j = torch.LongTensor(indices[i:min(i+batch_size, num_examples)])
# 		# long表示是64位float
# 		yield features.index_select(0, j), labels.index_select(0, j)
# 		# 第一个参数dim表示从第几维挑选数据，第二个为维度的哪个位置

# batch_size = 10

# # for X, y in data_iter(batch_size, features, labels):
# # 	print(X, y)
# # 	# print(X.dtype)
# # 	# print(y.dtype)
# # 	break


# '''初始化模型参数'''
# w = torch.tensor(np.random.normal(0, 0.01, (num_inputs, 1)),dtype=torch.float64)
# w.requires_grad_(requires_grad=True)
# b = torch.zeros(1, dtype=torch.float64)
# b.requires_grad_(requires_grad=True)

# '''定义模型'''
# def linreg(X, w, b):
# 	return torch.mm(X, w) + b
# 	# mm表示矩阵乘法

# '''定义损失函数'''
# def squared_loss(y_hat, y):
# 	return (y_hat -y.view(y_hat.size()))**2/2
# 	# view表示resize的操作，-1的作用相当于将矩阵展开为一个向量

# '''定义优化算法'''
# def sgd(params, lr, batch_size):
# 	for param in params:
# 		param.data -= lr*param.grad/batch_size

# '''训练模型'''
# # 迭代周期epoch
# # 需要对loss求和得到标量
# # 每次更新完后要将梯度清零
# lr = 0.03
# num_epochs = 3
# net = linreg
# loss = squared_loss
# for epoch in range(num_epochs):
# 	for X, y in data_iter(batch_size, features, labels):
# 		# X.to(torch.float32)
# 		l = loss(net(X, w, b), y).sum() # dtype必须要匹配
# 		l.backward()
# 		sgd([w, b], lr, batch_size)

# 		# 是在每个batch训练后，梯度清零
# 		w.grad.data.zero_()
# 		b.grad.data.zero_()

# 	train_l = loss(net(features,w,b), labels)
# 	print('epoch' + str(epoch + 1) + 'loss '+ str(train_l.mean().item()))


'''线性回归的简洁实现'''
# import torch.utils.data as Data 

# '''生成数据集'''
# num_inputs = 2
# num_examples = 1000
# true_w = [2, -3.4]
# true_b = 4.2
# features = torch.tensor(np.random.normal(0, 1, (num_examples, num_inputs)), dtype=torch.float)
# labels = true_w[0]*features[:, 0] + true_w[1]*features[:, 1] + true_b
# labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float)


# batch_size = 10 
# dataset = Data.TensorDataset(features, labels) # 将训练数据的特征和标签组合
# data_iter = Data.DataLoader(dataset, batch_size, shuffle=True) # 随机读取小批量


# # for X, y in data_iter:
# # 	# print(X, y)
# # 	print(X.dtype)
# # 	print(y.dtype)
# # 	break

# # nn.Linear in_features:输入样本的大小;out_features:输出样本的大小;bias:默认为True，可以设置为False，设置后则不会添加偏差
# # torch.nn仅支持输入一个batch的样本不支持单个样本输入
# class LinearNet(nn.Module):
# 	def __init__(self, n_feature):
# 		super(LinearNet, self).__init__()
# 		# 可表示一个或多个层的神经网络
# 		self.linear = nn.Linear(n_feature, 1)
# 	def forward(self, x):
# 		y = self.linear(x)
# 		return y

# # net = LinearNet(num_inputs)
# # print(net) # 打印出网络结构
		
# # 利用nn.Sequential搭建网络，传入其他层
# # 方法1
# net = nn.Sequential(nn.Linear(num_inputs, 1)) #此处可传入其他层
# # print(net[0])

# # # 方法2
# # net = nn.Sequential() #此处可传入其他层
# # net.add_module('linear', nn.Linear(num_inputs, 1))
# # # net.add_module ......
# # print(net)

# # # 方法3
# # from collections import OrderedDict
# # net = nn.Sequential(OrderedDict([('linear', nn.Linear(num_inputs, 1))
# # 	# ......
# # 	]))
# # print(net)

# # 可以通过net.parameters()查看模型所有可学习参数，返回一个生成器
# # for param in net.parameters():
# # 	print(param)
# # 	param.to(torch.float64)
# # 	print(param.dtype)


# '''初始化模型参数'''
# from torch.nn import init

# init.normal_(net[0].weight, mean=0., std=0.01)
# init.constant_(net[0].bias, val=0.) # 也可以直接修改bias的data
# # net[0].weight.to(torch.float64)
# # net[0].bias.to(torch.float64)
# # print(net[0].bias.dtype)

# '''定义损失函数'''
# # loss可以看做一个特殊层，为nn.Module的子类
# loss = nn.MSELoss()

# '''定义优化算法'''
# # torch.optim提供优化算法
# import torch.optim as optim

# optimizer = optim.SGD(net.parameters(), lr=0.03)
# # print(optimizer)

# # optimizer = optim.SGD([
# # 	# 如果对某个参数不指定学习率，就使用最外层的默认学习率
# # 	{'params': net[0].parameters()},
# # 	{'params': net[1].parameters(), 'lr':0.01}
# # 	], lr=0.03)

# # 调整学习率
# # for param_group in optimizer.param_groups:
# # 	param_group['lr'] *= 0.1


# '''训练模型'''
# num_epochs = 9
# for epoch in range(1, num_epochs+1):
# 	for X, y in data_iter:
# 		output = net(X)
# 		l = loss(output, y.view(-1, 1))
# 		optimizer.zero_grad() # 等于net.zero_grad()
# 		l.backward()
# 		optimizer.step()
# 	print('epoch %d, loss: %f' % (epoch, l.item()))

# dense = net[0]
# print(true_w, dense.weight)
# print(true_b, dense.bias)

'''可视化'''
# w1 = dense.weight.detach().numpy().reshape(-1, 1)
# b = dense.bias.detach().numpy()
# # y_pred = torch.mm(features, dense.weight.view(-1, 1)) + dense.bias
# # print(features[:, 1].numpy().reshape(-1).shape)
# # print(y_pred.detach().numpy().reshape(-1).shape)
# x1 = np.linspace(-3, 4, 100)[:,None]
# y_pred = w1[1]*x1 + b

# plt.scatter(features[:, 1].numpy(), labels.numpy(), 1)
# plt.plot(x1, y_pred)
# plt.show()






















