'''
这个脚本是pytorch可能会用到的函数
'''

import torch
import torch.nn as nn
from matplotlib import pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import sys

def linreg(X, w, b):
	return torch.mm(X, w) + b

def squared_loss(y_hat, y):
	return (y_hat - y.view(y_hat.size())) ** 2 / 2

def set_figsize(figsize=(3.5, 2.5)):
	plt.rcParams['figure.figsize'] = figsize 


def load_data_fashion_mnist(batch_size):
	
	minst_train = torchvision.datasets.FashionMNIST(
	root = 'E:\\DFIM\\McIntosh\\ML&DL\\FashionMNIST\\raw',
	train=True,
	download=False,
	transform=transforms.ToTensor())

	minst_test = torchvision.datasets.FashionMNIST(
	root = 'E:\\DFIM\\McIntosh\\ML&DL\\FashionMNIST\\raw',
	train=False,
	download=False,
	transform=transforms.ToTensor())

	if sys.platform.startswith('win'):
		num_workers = 0
	else:
		num_workers = 4

	train_iter = torch.utils.data.DataLoader(minst_train,
		batch_size=batch_size, shuffle=True, num_workers=num_workers)
	test_iter = torch.utils.data.DataLoader(minst_test,
		batch_size=batch_size, shuffle=True, num_workers=num_workers)

	return train_iter, test_iter

def train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size,
	params=None, lr=None, optimizer=None):
	for epoch in range(num_epochs):
		train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
		for X, y in train_iter:
			y_hat = net(X)
			l = loss(y_hat, y).sum()

			l.backward()
			if optimizer is None:
				sgd(params, lr, batch_size)
			else:
				optimizer.step()

			if optimizer is not None:
				optimizer.zero_grad()
			elif params is not None and params[0].grad is not None:
				for param in params:
					param.grad.data.zero_()

			train_l_sum += l.item()
			train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
			n += y.shape[0]
		
		# 计算训练的参数在测试集的准确率
		test_acc = evluate_accuracy(test_iter, net)
		print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f'
				 % (epoch+1, train_l_sum/n, train_acc_sum/n, test_acc))

def evluate_accuracy(data_iter, net):
	acc_sum, n = 0.0, 0
	for X, y in data_iter:
		if isinstance(net, torch.nn.Module):
			# isinstance判断一个对象是否是一个已知的类型，如果net和已知的后面一个类型，则返回true
			net.eval() # 评估模式，关闭dropout
			acc_sum += (net(X).argmax(dim=1)==y).float().sum().item()
			net.train() # 改回训练模式
		else:
			if('is_training' in net.__code__.co_varnames):
				# 获取函数内部变量名称，如果有is_training这个参数
				acc_sum += (net(X).argmax(dim=1)==y).float().sum().item()
		n += y.shape[0]
	return acc_sum / n


def sgd(params, lr, batch_size):
	for param in params:
		param.data -= lr*param.grad/batch_size



def get_fashion_mnist_labels(labels):
	text_labels = ['t-shirt', 'trouser', 'pullover', 'dress','coat',
	'sandal', 'shirt', 'sneaker', 'bag', 'ankleboot']
	return [text_labels[int(i)] for i in labels]

def show_fashion_minist(images, labels):
	_, figs = plt.subplots(1, len(images), figsize=(12, 12))
	# subplots返回一个图形对象和所有zxes对象即子图的坐标系，可用索引访问
	for f, img, lbl in zip(figs, images, labels):
		'''
		将多个序列（列表、元组、字典、集合、字符串以及 range() 区间构成的列表）
		压缩”成一个 zip 对象，即将这些序列中对应位置的元素重新组合，生成一个个新的元组
		当多个序列中元素个数不一致时，会以最短的序列为准进行压缩。
		'''
		f.imshow(img.view((28, 28)).numpy())
		# 变为带标量数据的图像，长宽为28像素，并且转换为numpy
		f.set_title(lbl)
		f.axes.get_xaxis().set_visible(False) # 隐藏坐标轴刻度
		f.axes.get_yaxis().set_visible(False)
	plt.show()

class FlattenLayer(nn.Module):
	def __init__(self):
		super(FlattenLayer, self).__init__()
	def forward(self, x):
		return x.view(x.shape[0], -1)

def semilogy(x_vals, y_vals, x_label, y_label, x2_vals=None, 
	y2_vals=None, legend=None, figsize=(3.5, 2.5)):
	set_figsize(figsize)
	plt.xlabel(x_label)
	plt.ylabel(y_label)
	plt.semilogy(x_vals, y_vals)
	if x2_vals and y2_vals:
		plt.semilogy(x2_vals, y2_vals, linestyle=':') # ：表示线类型为散点
		plt.legend(legend) # 表示画出图例
	plt.show()


# def fit_and_plot(train_features, test_features, train_labels, test_labels):
# 	# 定义神经网络结构
# 	net = torch.nn.Linear(train_features.shape[-1], 1)
# 	# 不用手动初始化

# 	batch_size = min(10, train_labels.shape[0])
# 	dataset = torch.utils.data.TensorDataset(train_features, train_labels)
# 	train_iter = torch.utils.data.DataLoader(dataset,
# 		batch_size=batch_size, shuffle=True)

# 	optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
# 	train_ls, test_ls = [], []

# 	# 训练
# 	for _ in range(num_epochs):
# 		for X, y in train_iter:
# 			l = loss(net(X), y.view(-1, 1))
# 			optimizer.zero_grad()
# 			l.backward()
# 			optimizer.step()
# 		train_labels = train_labels.view(-1, 1)
# 		test_labels = test_labels.view(-1, 1)
# 		train_ls.append(loss(net(train_features),
# 			train_labels).item())
# 		test_ls.append(loss(net(test_features),
# 			test_labels).item())
# 		print('final epoch: train loss', train_ls[-1], 'test loss', test_ls[-1])
# 		semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'loss',
# 			range(1, num_epochs + 1), test_ls, ['train', 'test'])
# 		print('weight:', net.weight.data, '\nbias:', net.bias.data)

def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
	'''设置matplotlib的轴'''
	axes.set_xlabel(xlabel)
	axes.set_ylabel(ylabel)
	axes.set_xscale(xscale) # 设置坐标轴的类型
	axes.set_yscale(yscale)
	axes.set_xlim(xlim)
	axes.set_ylim(ylim)
	if legend:
		axes.legend(legend)
	axes.grid()

def plot(X, Y=None, xlabel=None, ylabel=None, legend=None, xlim=None, 
	ylim=None, xscale='linear', yscale='linear',
	fmts=('-', 'm--', 'g-.', 'r:'), figsize=(3.5, 2.5), axes=None):
	if legend is None:
		legend = []

	set_figsize(figsize)
	# if axes:
	# 	axes = axes
	# else:
	# 	plt.gca()
	axes = axes if axes else plt.gca()

	# 如果X有⼀个轴，输出True
	def has_one_axis(X):
		return (hasattr(X, "ndim") and X.ndim == 1 or isinstance(X, list)
		and not hasattr(X[0], "__len__"))
	if has_one_axis(X):
		X = [X]
	if Y is None:
		X, Y = [[]] * len(X), X
	elif has_one_axis(Y):
		Y = [Y]
	if len(X) != len(Y):
		X = X * len(Y)
	axes.cla()
	for x, y, fmt in zip(X, Y, fmts):
		if len(x):
			axes.plot(x, y, fmt)
		else:
			axes.plot(y, fmt)
	set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
	plt.show()



	
