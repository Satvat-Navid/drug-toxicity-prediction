import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import rdkit
import fastapi


df = pd.read_csv("tox21.csv")

print(df.shape)
