import pandas as pd

df = pd.read_csv("data/final_dataset.csv")
print(df.head())
print(df.shape)
print(df.isnull().sum())
print(df.dtypes)