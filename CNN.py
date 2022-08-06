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
# ToTensor将图像转换为Tensor
# Normalize归一化处理，第一组为mean，第二组为std

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
	# 图片转换为tensor和归一化后，0的地方较多，所以调量一点，相当于归一化的逆向
	npimg = img.numpy() # 转换为numpy
	plt.imshow(np.transpose(npimg, (1,2,0)))
	# transpose指定参数后，就是把第一维放到第0维，第2维放到第1维，第0维放到第2维
	# 即由之前先绘制一张图所有点再绘制下一个通道，变为先绘制一个点的所有通道
	plt.show()

'''随机画出几张训练图像'''
# dataiter = iter(trainloader)
# images, labels = dataiter.next()

# # show images
# imshow(torchvision.utils.make_grid(images))
# # torchvision.utils.make_grid组合几张图片，padding默认为0，即图片的间距

# # print labels
# print(''.join(f'{classes[labels[j]]:5s} ' for j in range(batch_size)))

'''建立模型'''
class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		# 第一个参数是输入图像通道数，第二个参数6是输出通道，第三个参数5x5是卷积核大小
		# kernel 核的定义
		self.conv1 = nn.Conv2d(3, 6, 5)
		self.pool = nn.MaxPool2d(2,2)
		self.conv2 = nn.Conv2d(6, 16, 5)

		# 以下是全连接层
		self.fc1 = nn.Linear(16*5*5, 120)
		self.fc2 = nn.Linear(120, 84)
		self.fc3 = nn.Linear(84, 10)

	def forward(self, x):
		x = self.pool(F.relu(self.conv1(x)))
		x = self.pool(F.relu(self.conv2(x)))
		x = torch.flatten(x, 1)
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
net = Net()
net.to(device)

'''训练'''
import torch.optim as optim 

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(2):
	running_loss = 0
	for i, data in enumerate(trainloader, 0):
		# enumerate遍历对象所有元素和当前元素位置，0表示循环初始索引
		# inputs, labels = data
		inputs, labels = data[0].to(device), data[1].to(device)

		optimizer.zero_grad()
		output = net(inputs)
		loss = criterion(output, labels)
		loss.backward()
		optimizer.step()

		running_loss += loss.item()
		if i % 2000 == 1999:
			print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.10f}')
			running_loss = 0.0


'''将训练好的模型保存'''
# 保存模型参数
# PATH = './cifar_net.pth'
# print(net.state_dict()) # Net参数值保存在state_dict（状态字典）属性中
# torch.save(net.state_dict(), PATH)


'''测试'''
# dataiter = iter(testloader)
# images, labels = dataiter.next()

# 打印一些测试的图片
# imshow(torchvision.utils.make_grid(images))
# print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))

# net = Net()
# net.load_state_dict(torch.load(PATH))
# outputs = net(images)

# _, predicted = torch.max(outputs, 1)
# torch.max表示从output中按行求最大值，返回最大值和其索引
# dim=0，表示按列求最大值，并返回最大值的索引

# print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}' for j in range(4)))

'''用于整个测试集'''
# correct = 0
# total = 0

# with torch.no_grad():
# 	for images, labels in testloader:
# 		outputs = net(images)
# 		_, predicted = torch.max(outputs, 1)
# 		total += labels.size(0) # 一共有多少个图片
# 		correct += (predicted == labels).sum().item()

# # print(total)
# print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

'''network在哪些方面做的好，哪些方面不好'''
# correct_pred = {classname: 0 for classname in classes}
# total_pred = {classname: 0 for classname in classes}
# # print(correct_pred)
# # print(total_pred)

# with torch.no_grad():
# 	for data in testloader:
# 		images, labels = data
# 		outputs = net(images)
# 		_, predicted = torch.max(outputs, 1)

# 		# 给每个分类收集正确的预测
# 		for label, prediction in zip(labels, predicted):
# 			if label == prediction:
# 				correct_pred[classes[label]] += 1
# 			total_pred[classes[label]] += 1

# for classname, correct_count in correct_pred.items():
# 	accuracy = 100*float(correct_count) / total_pred[classname]
# 	print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')

'''在GPU上训练'''
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# print(device)

# net.to(device)

# inputs, labels = data[0].to(device), data[1].to(device)