import os
import pandas as pd

from sklearn.model_selection import train_test_split

# first of all we need to remove missing values rows, because ML hates missing values while
# supervised(Labelled) learning, and that's why we analyse and delete the those rows usingdropna().
# And we have to do this before splitting the data into training and testing sets, because if we split the data first, we might end up with missing values in the training set or testing set, which will cause problems while training the model or evaluating the model.

# to get the full path of 'Dataset.csv'
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, 'Dataset.csv')

df = pd.read_csv(csv_path)

print("rows and columns:", df.shape, "\n")

print(df.isnull().sum())   # to check the number of rows with missing values

df = df.dropna()   # to remove rows with missing values

print("\n", df.shape)   # to check the number of rows and columns in the Dataset after removing missing values

# Now we have a clean dataset without missing values, we can split the data into training and testing sets using train_test_split() function from sklearn library.

features = ["Votes", "Price range", "Average Cost for two"] # selecting needed columns for training

X = df[features]    # Inputs
y = df["Aggregate rating"]    # Output

X_train, X_test, y_train, y_test = train_test_split(
  X,
  y,
  test_size = 0.2,   # 20% of the data will be used for testing and 80% for training
  random_state = 42
)

