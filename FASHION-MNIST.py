'''图像分类数据集 FASHION-MNIST'''
import torch
import torchvision
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
import numpy as np 
import time
import sys


minst_train = torchvision.datasets.FashionMNIST(
	root = 'E:\\DFIM\\McIntosh\\ML&DL\\FashionMNIST\\raw',
	train=True,
	download=False,
	transform=transforms.ToTensor()
	)
# minst_train是torch.utils.data.Dataset的子集

'''
下载数据集，通过参数train指定获取训练集或测试集
transform = transforms.ToTensor() 使所有数据转换为tensor，
即将尺寸为(H x W x C)的图片且数据在[0, 255]的PIL图片
或数据类型为np.uint8的numpy数组转换为尺寸为(C x H x W)且数据类型为
torch.float32且位于[0.0, 1.0]的tensor
'''

minst_test = torchvision.datasets.FashionMNIST(
	root = 'E:\\DFIM\\McIntosh\\ML&DL\\FashionMNIST\\raw',
	train=False,
	download=False,
	transform=transforms.ToTensor()
	)

# 每个类别的图像分别为6000和1000，一共10个类别
# print(type(minst_train))
# print(len(minst_train), len(minst_test))

# print(minst_train[0]) # (tensor, 1个number)的组合

feature, label = minst_train[0]
# print(feature.shape, label)
# 宽高均为28像素，通道数为1，是灰度图像


# 定义一个函数，使label数值变成文本标签
def get_fashion_mnist_labels(labels):
	text_labels = ['t-shirt', 'trouser', 'pullover', 'dress','coat',
	'sandal', 'shirt', 'sneaker', 'bag', 'ankleboot']
	return [text_labels[int(i)] for i in labels]

# 创建一个函数可以在一行画出多张图像和对应标签的函数
# images是一个list，元素时image
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

# X, y =[], []
# for i in range(10):
# 	X.append(minst_train[i][0]) # 图像
# 	y.append(minst_train[i][1]) # 标签
# show_fashion_minist(X, get_fashion_mnist_labels(y))


'''读取小批量'''
batch_size = 256
if sys.platform.startswith('win'):
	num_workers = 0
else:
	num_workers = 4

train_iter = torch.utils.data.DataLoader(minst_train,
	batch_size=batch_size, shuffle=True, num_workers=num_workers)
test_iter = torch.utils.data.DataLoader(minst_test,
	batch_size=batch_size, shuffle=True, num_workers=num_workers)

# start = time.time()
# for X, y in train_iter:
# 	continue
# print('%.2f sec' % (time.time() - start))

'''初始化模型参数'''
num_inputs = 784
num_outputs = 10

# w的size是784 x 10
# b的size是1 x 10
# W = torch.tensor(np.random.normal(0, 0.01, (num_inputs, num_outputs)), dtype=torch.float)
# b = torch.zeros(num_outputs, dtype=torch.float)

# W.requires_grad_(requires_grad=True)
# b.requires_grad_(requires_grad=True)

# # softmax
# # x行数为样本数，列数为输出个数，对同行求和，再除以和，每行元素和为1且非负
# # 所以每行可以看做概率分布，一个样本在各个输出2类别上的预测概率
# def softmax(X):
# 	X_exp = X.exp()
# 	partition = X_exp.sum(dim=1, keepdim=True)
# 	return X_exp / partition # 应用了广播机制

# # X = torch.rand((2, 5))
# # X_prob = softmax(X)
# # print(X_prob, X_prob.sum(dim=1))

# '''定义模型'''
# def net(X):
# 	return softmax(torch.mm(X.view((-1, num_inputs)), W) + b)

# '''定义损失函数'''
# def cross_entropy(y_hat, y):
# 	return - torch.log(y_hat.gather(1, y.view(-1, 1)))


# '''定义分类准确率'''
def evluate_accuracy(data_iter, net):
	acc_sum, n = 0, 0
	for X, y in data_iter:
		acc_sum += (net(X).argmax(dim=1)==y).float().sum().item()
		n += y.shape[0]
	return acc_sum / n

# # print(evluate_accuracy(test_iter, net))
# # net只需输入函数名，同时net内部只有X一个局部变量

# '''定义优化算法'''
# def sgd(params, lr, batch_size):
# 	for param in params:
# 		param.data -= lr*param.grad/batch_size

# '''训练模型'''
# params = [W, b]
# num_epochs, lr = 5, 0.1
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

# train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, batch_size, [W, b], lr)

# '''预测'''
# X, y = iter(test_iter).next()

# # tensor要转换为numpy数组
# true_labels = get_fashion_mnist_labels(y.numpy())
# pred_labels = get_fashion_mnist_labels(net(X).argmax(dim=1).numpy())
# titles = [true + '\n' + pred for true, pred in zip(true_labels, pred_labels)]

# show_fashion_minist(X[0:10], titles[0:10])



'''
简洁实现
'''

from torch import nn
from torch.nn import init


# 每个batch的x形状为(batch_size, 1, 28, 28),要把shape转换为(batch_size, 784)

# class LinearNet(nn.Module):
# 	def __init__(self, num_inputs, num_outputs):
# 		super(LinearNet, self).__init__():
# 		self.linear = nn.Linear(num_inputs, num_outputs)
# 	def forward(self, x):
# 		y = self.linear(x.view(x.shape[0], -1))

# net = LinearNet(num_inputs, num_outputs)

# 可以将对x形状转换功能定义为一个FlattenLayer
class FlattenLayer(nn.Module):
	def __init__(self):
		super(FlattenLayer, self).__init__()
	def forward(self, x):
		return x.view(x.shape[0], -1)

net = nn.Sequential(FlattenLayer(), nn.Linear(num_inputs, num_outputs))

# 初始化参数
init.normal_(net[1].weight, mean=0., std=0.01)
init.constant_(net[1].bias, val=0.)


'''定义损失函数'''
loss = nn.CrossEntropyLoss()

'''定义优化算法'''
optimizer = torch.optim.SGD(net.parameters(), lr=0.1)

'''训练'''
num_epochs = 5
train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, None, None, optimizer)






'''tensor求和'''
# X = torch.tensor([[1, 2, 3], [4, 5, 6]])
# print(X.sum(dim=0, keepdim=True)) # tensor([[5, 7, 9]])；FLASE：tensor([5, 7, 9])
# # keepdim保留维度，False相当于求和后这个维度就没了，但是True就保存
# print(X.sum(dim=1, keepdim=True)) # tensor([[ 6],[15]])；false：tensor([ 6, 15])


'''广播机制'''
# x = torch.arange(1,3),view(1,2) # tensor([[1, 2]]) 自动增加一个维度,便于广播机制
# print(torch.arange(1,3)) # tensor([1, 2])
# y = torch.arange(1, 4).view(3, 1) # tensor([[1],[2],[3]])
# print(torch.arange(1, 4)) # tensor([1, 2, 3])
# print(x + y) # tensor([[2, 3],[3, 4],[4, 5]])
# # x的第一行两个元素复制到2、3行，y的第一列3个元素复制到第二列

'''gather函数'''
# x.gather(dim, index)
# dim表示索引的方式，dim=1说明索引列号，即横向取值；dim=0，纵向取值

# a = torch.Tensor([[1,2,3],[4,5,6]])
# index = torch.LongTensor([[0,1],[2,0]]) # index必为long
# print(a.gather(0,index))
# 输出
# tensor([[1., 2.],
#         [6., 4.]])
# index第一行在tensor第一行取第0个和第1个

# index = torch.LongTensor([[0,1,1],[0,0,1]])
# print(a.gather(0,index))
# 输出
# tensor([[1., 5., 6.],
#         [1., 2., 6.]])
# index第二列在tensor第一列取第1个和第0个
# 在分类问题的应用
# y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]]) # 2个样本3个类别预测值
# y = torch.LongTensor([0, 2]) # 标签值
# y_hat.gather(1, y.view(-1, 1)) # 取标签值对应的输出

# 准确率函数
# def accuracy(y_hat, y):
# 	return((y_hat.argmax(dim=1)==y).float().mean().item())
	# argmax取每行即dim=1的最大值的索引，刚好等于对应标签值。
	# 如果是对应的标签等于真实的标签则为true，并转换为float
	# 取平均值，不过此时结果还是tensor，要用item取出里面的值