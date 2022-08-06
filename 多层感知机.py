'''
多层感知机，就是加一层隐藏层
'''

import torch
import numpy as np 
from matplotlib import pyplot as plt
import sys
import d2lzh_pytorch as d2l
from torch import nn
from torch.nn import init

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

'''定义模型参数'''
num_inputs, num_outputs, num_hiddens = 784, 10, 256

# W1 = torch.tensor(np.random.normal(0, 0.01, (num_inputs, num_hiddens)), dtype=torch.float)
# b1 = torch.zeros(num_hiddens, dtype=torch.float)
# W2 = torch.tensor(np.random.normal(0, 0.01, (num_hiddens, num_outputs)), dtype=torch.float)
# b2 = torch.zeros(num_outputs, dtype=torch.float)

# params = [W1, b1, W2, b2]
# for param in params:
# 	param.requires_grad_(requires_grad=True)


# '''定义激活函数'''
# def relu(X):
# 	return torch.max(input=X, other=torch.tensor(0.0))


# '''定义模型'''
# def net(X):
# 	X = X.view((-1, num_inputs))
# 	H = relu(torch.matmul(X, W1) + b1)
# 	return torch.matmul(H, W2) + b2

# '''定义损失函数'''
# loss = torch.nn.CrossEntropyLoss()

# '''训练模型'''
# # 因为除以batch size所以梯度很小，所以lr翻倍
# num_epochs, lr =5, 100.0
# d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, params, lr)



'''
简洁实现
'''
net = nn.Sequential(
	d2l.FlattenLayer(), 
	nn.Linear(num_inputs, num_hiddens),
	nn.ReLU(),
	nn.Linear(num_hiddens, num_outputs),
	)

for param in net.parameters():
	init.normal_(param, mean=0., std=0.01)

loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.5)

num_epochs = 5
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, None, None, optimizer)















































'''pytorch中使用激活函数'''

# def xyplot(x_vals, y_vals, name):
# 	d2l.set_figsize(figsize=(5, 2.5))
# 	plt.plot(x_vals.detach().numpy(), y_vals.detach().numpy())
# 	# detach()返回一个新的tensor，从当前计算图中分离下来的,但是仍指向原变量的存放位置
# 	# requires_grad为false，即以后不计算其梯度
# 	plt.xlabel('x')
# 	plt.ylabel(name + '(x)')
# 	plt.show()

# x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)

'''relu函数及其导数的图像'''
# y = x.relu()
# xyplot(x, y, 'relu')

# y.sum().backward()
# xyplot(x, x.grad, 'grad of relu')

'''sigmoid函数及其导数的图像'''
# y = x.sigmoid()
# # xyplot(x, y, 'sigmoid')

# # x.grad.zero_()
# y.sum().backward()
# xyplot(x, x.grad, 'grad of sigmoid')

'''tanh函数及其导数的图像'''
# y = x.tanh()
# # xyplot(x, y, 'tanh')

# # x.grad.zero_()
# y.sum().backward()
# xyplot(x, x.grad, 'grad of tanh')


# num_inputs = 784
# num_outputs = 10
# class FlattenLayer(nn.Module):
# 	def __init__(self):
# 		super(FlattenLayer, self).__init__()
# 	def forward(self, x):
# 		return x.view(x.shape[0], -1)

# net = nn.Sequential(FlattenLayer(), nn.Linear(num_inputs, num_outputs))

# # 初始化参数
# init.normal_(net[1].weight, mean=0., std=0.01)
# init.constant_(net[1].bias, val=0.)


# '''定义损失函数'''
# loss = nn.CrossEntropyLoss()

# '''定义优化算法'''
# optimizer = torch.optim.SGD(net.parameters(), lr=0.1)

# '''训练'''
# num_epochs = 5
# d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, None, None, optimizer)

# '''预测'''
# X, y = iter(test_iter).next()
# true_labels = d2l.get_fashion_mnist_labels(y.numpy())
# pred_labels = d2l.get_fashion_mnist_labels(net(X).argmax(dim=1).numpy())
# titles = [true + '\n' + pred for true, pred in zip(true_labels, pred_labels)]

# d2l.show_fashion_minist(X[0:10], titles[0:10])