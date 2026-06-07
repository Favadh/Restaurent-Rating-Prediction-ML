import os
import pandas as pd

script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, 'Dataset.csv')

print('Resolved CSV path:', csv_path)

df = pd.read_csv(csv_path)
print(df.head())