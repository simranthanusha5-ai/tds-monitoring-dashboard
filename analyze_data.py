import pandas as pd

file_path = "data/PrimeOxy TDS Dataset.xlsx"

data = pd.read_excel(file_path)

print(data.head())
print(data.shape)
print(data.columns)
print(data.isnull().sum())