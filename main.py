import os
import pandas as pd

# first of all we need to remove missing values rows, because ML hates missing values while
# supervised(Labelled) learning, and that's why we analyse and understand dataset using shape, columns,
# info(), isnull.sum(), dropna().
# And we have to do this before splitting the data into training and testing sets, because if we split the data first, we might end up with missing values in the training set or testing set, which will cause problems while training the model or evaluating the model.

# to get the full path of 'Dataset.csv'
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, 'Dataset.csv')

print('Resolved CSV path:', csv_path)

df = pd.read_csv(csv_path)
print(df.shape)   # to check the number of rows and columns in the Dataset
print(df.columns)   # to check the column names in the Dataset
print(df.info())    # to check the data types of each column and if there are any missing values

print(df.isnull().sum())   # to check the number of missing values in each column

df = df.dropna()   # to remove rows with missing values

print(df.shape)   # to check the number of rows and columns in the Dataset after removing missing values