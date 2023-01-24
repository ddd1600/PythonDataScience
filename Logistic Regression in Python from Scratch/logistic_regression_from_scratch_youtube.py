import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

X_train = pd.read("train_X.csv").drop("Id", axis=1).values
Y_train = pd.read("train_Y.csv").drop("Id", axis=1).values
X_test  = pd.read("test_X.csv").drop("Id", axis=1).values
Y_test  = pd.read("test_X.csv").drop("Id", axis=1).values