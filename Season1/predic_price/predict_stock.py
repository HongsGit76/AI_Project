import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense, Activation

# from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau
import datetime

data = pd.read_csv('C:/Users/sdat7/Desktop/1일1커밋/some project/predic_price/005930.KS.csv')
print(data.head())