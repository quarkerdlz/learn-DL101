'''多项式拟合实验'''

import torch
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt
import sys
import d2lzh_pytorch as d2l

'''生成多项式数据集'''
# 给定样本特征x，用以下函数生成样本的标签
# n_train, n_test, true_w, true_b = 100, 100, [1.2, -3.4, 5.6], 5
# features = torch.randn((n_train + n_test, 1))
# poly_features = torch.cat((features, torch.pow(features, 2), torch.pow(features, 3)), 1)
# labels = (true_w[0]*poly_features[:, 0] + true_w[1]*poly_features[:, 1]
# 	+ true_w[2]*poly_features[:, 2] + true_b)
# labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float)

# # print(features[:2]),
# # print(poly_features[:2])
# # print(labels[:2])

# num_epochs, loss = 100, torch.nn.MSELoss()

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
# 	print('final epoch: train loss', train_ls[-1], 'test loss', test_ls[-1])
# 	d2l.semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'loss',
# 		range(1, num_epochs + 1), test_ls, ['train', 'test'])
# 	print('weight:', net.weight.data, '\nbias:', net.bias.data)

# # fit_and_plot(poly_features[:n_train, :], poly_features[n_train:, :],
# # 	labels[:n_train], labels[n_train:])

# '''如果使用线性拟合(欠拟合）'''
# # 使用的是线性模型
# # fit_and_plot(features[:n_train, :], features[n_train:, :],
# # 	labels[:n_train], labels[n_train:])
# # 误差难以下降

# '''过拟合'''
# # 使训练数据量变少，小于验证集，甚至小于参数数量
# fit_and_plot(poly_features[0:25, :], poly_features[n_train:, :],
# 	labels[0:25], labels[n_train:])

# 数据太少，导致过拟合。显得模型过于复杂，容易受到训练中数据噪声的影响

'''权重衰减方法'''
# 高维线性回归实验

# 定义一个高维线性模型
n_train, n_test, num_inputs = 20, 100, 200
true_w, true_b = torch.ones(num_inputs, 1) *0.01, 0.05

features = torch.randn((n_train+n_test, num_inputs))
labels = torch.matmul(features, true_w) + true_b
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float)

train_features, test_features = features[:n_train, :], features[n_train:, :]
train_labels, test_labels = labels[:n_train], labels[n_train:]

# 用底层方法
# 初始化模型参数
# def init_params():
# 	w = torch.randn((num_inputs, 1), requires_grad=True)
# 	b = torch.zeros(1, requires_grad=True)
# 	return [w, b]

# # 定义L2范数惩罚项
# def l2_penalty(w):
# 	return (w**2).sum() / 2

# # 定义训练和测试
batch_size, num_epochs, lr = 1, 100, 0.003
# net, loss = d2l.linreg, d2l.squared_loss

dataset = torch.utils.data.TensorDataset(train_features, train_labels)
train_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)

# def fit_and_plot(lambd):
# 	w, b = init_params()
# 	train_ls, test_ls = [], []

# 	# 训练
# 	for _ in range(num_epochs):
# 		for X, y in train_iter:
# 			l = loss(net(X, w, b), y) + lambd*l2_penalty(w)
# 			l = l.sum()

# 			if w.grad is not None:
# 				w.grad.data.zero_()
# 				b.grad.data.zero_()
# 			l.backward()
# 			d2l.sgd([w, b], lr, batch_size)

# 		train_ls.append(loss(net(train_features, w, b), train_labels).mean().item()) # 只有scalar可以用item
# 		test_ls.append(loss(net(test_features, w, b), test_labels).mean().item())
	
# 	d2l.semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'loss',
# 		range(1, num_epochs + 1), test_ls, ['train', 'test'])
# 	print('L2 norm of w:', w.norm().item())

# # fit_and_plot(lambd=0) # 典型过拟合图像

# # 使用权重衰减
# fit_and_plot(lambd=10)

# 简洁实现
# loss = d2l.squared_loss
# def fit_and_plot_pytorch(wd):
# 	net = nn.Linear(num_inputs, 1)
# 	nn.init.normal_(net.weight, mean=0, std=1)
# 	nn.init.normal_(net.bias, mean=0, std=1)
# 	optimizer_w = torch.optim.SGD(params=[net.weight], lr=lr, weight_decay=wd)
# 	optimizer_b = torch.optim.SGD(params=[net.bias], lr=lr) # 不对偏差参数衰减

# 	train_ls, test_ls = [], []

# 	# 训练
# 	for _ in range(num_epochs):
# 		for X, y in train_iter:
# 			l = loss(net(X), y).mean()
# 			optimizer_w.zero_grad()
# 			optimizer_b.zero_grad()

# 			l.backward()
# 			optimizer_w.step()
# 			optimizer_b.step()

# 		train_ls.append(loss(net(train_features), train_labels).mean().item()) # 只有scalar可以用item
# 		test_ls.append(loss(net(test_features), test_labels).mean().item())
	
# 	d2l.semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'loss',
# 		range(1, num_epochs + 1), test_ls, ['train', 'test'])
# 	print('L2 norm of w:', net.weight.data.norm().item())

# fit_and_plot_pytorch(5)


'''抓爆法'''
# 抓爆法用以抑制过拟合
def dropout(X, drop_prob):
	X = X.float()
	assert 0 <= drop_prob <= 1 #assert为判断语句
	keep_prob = 1 - drop_prob

	if keep_prob == 0:
		return torch.zeros_like(X)
	if keep_prob == 1:
		return X

	mask = (torch.rand(X.shape) < keep_prob).float() #选出小于keep的元素并转换为1
	# randn是均值为0，方差为1的正态分布
	# rand是[0,1)均匀分布
	return mask * X / keep_prob


# 定义模型参数
# 两个隐藏层
num_inputs, num_outputs, num_hiddens1, num_hiddens2 = 784, 10, 256, 256

# W1 = torch.tensor(np.random.normal(0, 0.01, (num_inputs, num_hiddens1)),dtype=torch.float, requires_grad=True)
# b1 = torch.zeros(num_hiddens1, dtype=torch.float, requires_grad=True)
# W2 = torch.tensor(np.random.normal(0, 0.01, (num_hiddens1, num_hiddens2)),dtype=torch.float, requires_grad=True)
# b2 = torch.zeros(num_hiddens2, dtype=torch.float, requires_grad=True)
# W3 = torch.tensor(np.random.normal(0, 0.01, (num_hiddens2, num_outputs)),dtype=torch.float, requires_grad=True)
# b3 = torch.zeros(num_outputs, dtype=torch.float, requires_grad=True)

# params = [W1, b1, W2, b2, W3, b3]

# 定义模型
drop_prob1, drop_prob2 = 0.2, 0.5

# def net(X, is_training=True):
# 	X = X.view(-1, num_inputs)
# 	H1 = (torch.matmul(X, W1) + b1).relu()
# 	if is_training:
# 		H1 = dropout(H1, drop_prob1)
# 	H2 = (torch.matmul(H1, W2) + b2).relu()
# 	if is_training:
# 		H2 = dropout(H2, drop_prob2)
# 	return torch.matmul(H2, W3) + b3

# 测试模型时不用抓爆
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
				acc_sum += (net(X, is_training=False).argmax(dim=1)==y).float().sum().item()
			else:
				acc_sum += (net(X).argmax(dim=1)==y).float().sum().item()
		n += y.shape[0]
	return acc_sum / n

# num_epochs, lr, batch_size = 5, 100.0, 256
# loss = torch.nn.CrossEntropyLoss()

# train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
# d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, params, lr)

# 简单实现
# net = nn.Sequential(
# 	d2l.FlattenLayer(),
# 	nn.Linear(num_inputs, num_hiddens1),
# 	nn.ReLU(),
# 	nn.Dropout(drop_prob1),
# 	nn.Linear(num_hiddens1, num_hiddens2),
# 	nn.ReLU(),
# 	nn.Dropout(drop_prob2),
# 	nn.Linear(num_hiddens2, num_outputs),
# 	)

# for param in net.parameters():
# 	nn.init.normal_(param, mean=0, std=0.01)

# optimizer = torch.optim.SGD(net.parameters(), lr=0.5)
# d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, None, None, optimizer)


'''定义模型的第二种方法'''
class Net(nn.Module):
	def __init__(self, num_inputs, num_outputs, num_hiddens1, num_hiddens2, is_training=True):
		super(Net, self).__init__()
		self.num_inputs = num_inputs
		self.training = is_training
		self.lin1 = nn.Linear(num_inputs, num_hiddens1)
		self.lin2 = nn.Linear(num_hiddens1, num_hiddens2)
		self.lin3 = nn.Linear(num_hiddens2, num_outputs)
		self.relu = nn.ReLU()

	def forward(self, X):
		H1 = self.relu(self.lin1(X.reshape((-1, self.num_inputs))))
		if self.training == True:
			H1 = dropout(H1, drop_prob1)
		H2 = self.relu(self.lin2(H1))
		if self.training == True:
			H2 = dropout(H2, drop_prob2)
		out = self.lin3(H2)
		return out

net = Net(num_inputs, num_outputs, num_hiddens1, num_hiddens2)

num_epochs, lr, batch_size = 10, 0.5, 256
loss = torch.nn.CrossEntropyLoss()

train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
trainer = torch.optim.SGD(net.parameters(), lr=lr)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, None, None, trainer)
