'''CNN的简单实现'''

import torch
import torch.nn as nn
import torch.nn.functional as F

# 建立CNN结构
# class Net(nn.Module):
# 	def __init__(self):
# 		super(Net, self).__init__()
# 		# 第一个参数是输入图像通道数，第二个参数6是输出通道，第三个参数5x5是卷积核大小
# 		# kernel 核的定义
# 		self.conv1 = nn.Conv2d(1, 6, 5)
# 		self.conv2 = nn.Conv2d(6, 16, 5)

# 		# 以下是全连接层
# 		self.fc1 = nn.Linear(16*5*5, 120)
# 		self.fc2 = nn.Linear(120, 84)
# 		self.fc3 = nn.Linear(84, 10)

# 	def forward(self, x):
# 		# max pooling做(2,2)的window
# 		x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
# 		# 如果size是方阵，可以用单个数
# 		x = F.max_pool2d(F.relu(self.conv2(x)), 2)
# 		x = torch.flatten(x, 1)
# 		x = F.relu(self.fc1(x))
# 		x = F.relu(self.fc2(x))
# 		x = self.fc3(x)
# 		return x

# net = Net()
# # print(net)

# # params = list(net.parameters())
# # print(len(params))
# # print(params[1])
# # print(params.size()) 
# # print(params[0].size()) # conv1的w

# input = torch.randn(1, 1, 32, 32)
# # out = net(input)
# # print(out)

# # 初始化梯度
# # net.zero_grad()
# # print('conv1.bias.grad before backward')
# # print(net.conv1.bias.grad)

# # out.backward(torch.randn(1, 10))

# # 设定损失函数
# # output = net(input)
# # target = torch.randn(10)
# # target = target.view(1,-1)
# criterion = nn.MSELoss()

# # loss = criterion(out, target)
# # print(loss)

# # print(loss.grad_fn)
# # print(loss.grad_fn.next_functions[0][0])
# # loss.backward()

# # print('conv1.bias.grad after backward')
# # print(net.conv1.bias.grad)

# # def print_graph(g, level=0):
# #     if g == None: return
# #     print('*'*level*4, g)
# #     for subg in g.next_functions:
# #         print_graph(subg[0], level+1)

# # print_graph(loss.grad_fn, 0)

# # 更新参数
# import torch.optim as optim

# optimizer = optim.SGD(net.parameters(), lr=0.01)

# optimizer.zero_grad()
# out = net(input)
# loss = criterion(out, target)
# loss.backward()
# optimizer.step()


'''实战'''
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose(
	[transforms.ToTensor(), 
	transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))])
# Compose串联多个transforms操作
# 

batch_size = 4

trainset = torchvision.datasets.CIFAR10(root='E:\\DFIM\\McIntosh\\ML&DL\\data1', train=True,
	download=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
	shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(root='E:\\DFIM\\McIntosh\\ML&DL\\data1', train=False, 
	download=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
	shuffle=True, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat', 
	'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

import matplotlib.pyplot as plt
import numpy as np

def imshow(img):
	img = img / 2 + 0.5
	npimg = img.numpy()
	plt.imshow(np.transpose(npimg, (1,2,0)))
	plt.show()

dataiter = iter(trainloader)
images, labels = dataiter.next()

# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print(''.join(f'{classes[labels[j]]:5s} ' for j in range(batch_size)))

'''建立模型'''
class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		# 第一个参数是输入图像通道数，第二个参数6是输出通道，第三个参数5x5是卷积核大小
		# kernel 核的定义
		self.conv1 = nn.Conv2d(1, 6, 5)
		self.conv2 = nn.Conv2d(6, 16, 5)

		# 以下是全连接层
		self.fc1 = nn.Linear(16*5*5, 120)
		self.fc2 = nn.Linear(120, 84)
		self.fc3 = nn.Linear(84, 10)

	def forward(self, x):
		# max pooling做(2,2)的window
		x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
		# 如果size是方阵，可以用单个数
		x = F.max_pool2d(F.relu(self.conv2(x)), 2)
		x = torch.flatten(x, 1)
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x

net = Net()

