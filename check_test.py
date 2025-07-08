import pickle
import pandas as pd

# Replace with your actual file path
file_path = '/Users/anmolsen/Documents/icpr2020/icpr2020dfdc/data/celebdf_faces_df.pkl'  # Change this to your actual .pkl file path

with open(file_path, 'rb') as f:
    data = pickle.load(f)

if isinstance(data, pd.DataFrame):
    total_rows = len(data)
    if 'test' in data.columns:
        # Check for True (bool) or 'true' (case-insensitive string)
        mask = data['test'].apply(lambda x: x is True or (isinstance(x, str) and x.lower() == 'true'))
        count = mask.sum()
        print(f"Number of rows with True or 'true' in the 'test' column: {count}")
        print(f"Total number of rows in the DataFrame: {total_rows}")
    else:
        print("No 'test' column found in the DataFrame.")
        print(f"Total number of rows in the DataFrame: {total_rows}")
else:
    print(f"Loaded object is not a DataFrame, it is {type(data)}.")
