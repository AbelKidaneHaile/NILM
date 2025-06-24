import pandas as pd
from tqdm import tqdm
import numpy as np

file_path = "full_data_house1.csv"
train_path = "train_data_house1.csv"
test_path = "test_data_house1.csv"
test_ratio = 0.3
chunk_size = 100_000  # Adjust based on memory

print("Counting total rows...")
# Get total number of rows (excluding header)
with open(file_path) as f:
    total_rows = sum(1 for _ in f) - 1

# Compute how many for train/test
n_test = int(total_rows * test_ratio)
n_train = total_rows - n_test

# Randomly shuffle indices for test rows
print("Generating random row assignments...")
all_indices = np.random.permutation(total_rows)
test_indices = set(all_indices[:n_test])
train_indices = set(all_indices[n_test:])

print("Splitting and writing chunks...")

# Prepare CSV writers with header
reader = pd.read_csv(file_path, chunksize=chunk_size, iterator=True)
first_chunk = next(reader)
first_chunk.iloc[0:0].to_csv(train_path, index=False)  # write header
first_chunk.iloc[0:0].to_csv(test_path, index=False)   # write header

# Start from beginning again
reader = pd.read_csv(file_path, chunksize=chunk_size, iterator=True)
start_index = 0

for chunk in tqdm(reader):
    end_index = start_index + len(chunk)
    row_indices = np.arange(start_index, end_index)

    # Split based on global row index
    mask_test = [i in test_indices for i in row_indices]
    mask_train = [not i for i in mask_test]

    chunk.iloc[mask_train].to_csv(train_path, mode='a', header=False, index=False)
    chunk.iloc[mask_test].to_csv(test_path, mode='a', header=False, index=False)

    start_index = end_index

print("Finished writing train/test CSVs.")
