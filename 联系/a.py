import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import matplotlib.pyplot as plt

data_path = '03_bike_predictor/bike-sharing-dataset/hour.csv'
rides = pd.read_csv(data_path)
rides.head() #  查看rides数据框的前几行
counts = rides['cnt'][:50] #  取出前50个数据
x = np.arange(len(counts)) #  生成x轴数据
y = np.array(counts) #  生成y轴数据
plt.figure(figsize=(20, 5)) #  绘制图形
plt.plot(x,y,'')
plt.xlabel('X') #  设置x轴标签
plt.ylabel('Y') #  设置Y轴标签为Y

x = torch.FloatTensor(np.arange(len(counts),dtype=float)) #  将x转换为torch张量并调整形状
y = torch.FloatTensor(np.array(counts, dtype=float)) #  将y转换为torch张量并调整形状

sz = 10

weights = torch.randn((1,sz),requires_grad=True) 
biases = torch.randn((sz),requires_grad=True) #  初始化权重和偏置
weights2 = torch.randn((sz,1),requires_grad=True) #  初始化权重

learning_rate = 0.00001 #  设置学习率
losses = [] #  初始化损失列表

x = x.view(50,-1)
y = y.view(50,-1) #  调整x和y的形状

for i in range(100000):
    hidden = x * weights + biases #  计算隐藏层
    hidden = torch.sigmoid(hidden) #  激活函数
    prediction = hidden.mm(weights2) #  计算预测值
    loss = torch.mean((prediction - y) ** 2) #  计算损失
    losses.append(loss.data.numpy())  #  将loss的值转换为numpy数组，并添加到losses列表中
    if i % 1000 == 0: #  每1000次迭代打印一次损失
        print(i, loss.data.numpy()) #  打印当前迭代次数和损失值
        print('loss:',loss) #  打印损失值
    
    
    loss.backward() #  反向传播计算梯度

    weights.data.add_(- learning_rate * weights.grad.data)
    biases.data.add_(- learning_rate * biases.grad.data)
    weights2.data.add_(- learning_rate * weights2.grad.data)

    weights.grad.data.zero_()
    biases.grad.data.zero_()
    weights2.grad.data.zero_()

plt.plot(losses)
plt.xlabel('Epochs') #  设置x轴标签
plt.ylabel('Loss') #  设置y轴标签
plt.show() #  显示损失曲线
