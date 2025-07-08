import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import matplotlib.pyplot as plt

data_path = "book_DeepLearning_in_PyTorch_Source/03_bike_predictor/bike-sharing-dataset/hour.csv"
rides = pd.read_csv(data_path)
count = rides['count'].values