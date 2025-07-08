import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import matplotlib.pyplot as plt

data_path = '03_bike_predictor/bike-sharing-dataset/hour.csv'
rides = pd.read_csv(data_path)
rides.head()
counts = rides['cnt'][:50]
x = np.arange(len(counts))
y = np.array(counts)
plt.figure(figsize=(10, 5))
plt.plot(x,y,'')
plt.xlabel('X') #  设置x轴标签
plt.ylabel('Y') #  设置Y轴标签为Y
plt.show()
