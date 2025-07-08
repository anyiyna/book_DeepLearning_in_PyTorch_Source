import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import matplotlib.pyplot as plt

data_path = '/Users/anyi/Documents/GitHub/book_DeepLearning_in_PyTorch_Source/03_bike_predictor/bike-sharing-dataset/hour.csv'
rides = pd.read_csv(data_path)
counts = rides['cnt'][:50]
x = np.arange(len(counts))
y = np.array(counts)
plt.figure(figsize=(10, 5))
plt.plot(x,y,'0-')
plt.xlabel('X')
plt.ylabel('Y')
