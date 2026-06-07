import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# first of all we need to remove missing values rows, because ML hates missing values while
# supervised(Labelled) learning, and that's why we analyse and delete the those rows usingdropna().
# And we have to do this before splitting the data into training and testing sets, because if we split the data first, we might end up with missing values in the training set or testing set, which will cause problems while training the model or evaluating the model.

# to get the full path of 'Dataset.csv', idh prethyekichonnum nokanda
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

model = LinearRegression()   # creating a Linear Regression algorithm based ML model

model.fit(X_train, y_train)   # training the model with the training data

predictions = model.predict(X_test)  # making predictions with the testing data

mse = mean_squared_error(y_test, predictions) # measures the predictions error with y_test(actual result)
r2 = r2_score(y_test, predictions) # Measures how well the model explains the ratings

# Evaluation:
if r2 >= 0.8 and mse <= 0.5:
    evaluation = "Excellent: the model explains most of the variance and has low prediction error."
elif r2 >= 0.6 and mse <= 1.0:
    evaluation = "Good: the model explains a meaningful portion of rating variance and prediction error is moderate."
elif r2 >= 0.4 and mse <= 2.0:
    evaluation = "Fair: the model has moderate predictive power but error is still noticeable."
else:
    evaluation = "Poor: the model does not explain the ratings well enough or prediction error is too high. Consider adding more features or using a different algorithm."

print("Evaluation:", evaluation)

results = X_test.copy()
results["Actual Aggregate rating"] = y_test
results["Predicted Aggregate rating"] = predictions

print("\nRaw test inputs with actual and predicted ratings:")
print(results.head())