import pickle
import pandas as pd

def check_pkl_file_and_save_csv(file_path, output_csv_path):
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        print(f"Loaded data type: {type(data)}")
        
        if isinstance(data, pd.DataFrame):
            print("Columns in DataFrame:", data.columns.tolist())
            print("\nFirst 5 rows:")
            print(data.head())
            # Save first 10 rows to CSV
            data.head(10).to_csv(output_csv_path, index=True)
            print(f"First 10 rows saved to {output_csv_path}")
        else:
            print("Data is not a DataFrame, cannot save to CSV.")
    except Exception as e:
        print(f"Error loading file: {e}")

# Example usage
file_path = '/Users/anmolsen/Documents/icpr2020/icpr2020dfdc/scripts/results/net-Xception_traindb-celebdf_face-scale_size-224_seed-0_last/celebdf_test.pkl'
output_csv_path = '/Users/anmolsen/Documents/icpr2020/icpr2020dfdc/data/test_first10.csv'
check_pkl_file_and_save_csv(file_path, output_csv_path)

