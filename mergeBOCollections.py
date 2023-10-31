import os
import pandas as pd

# Get all csv files starting with 'box_office_collection_'
csv_files = [file for file in os.listdir() if file.startswith('box_office_collection_') and file.endswith('.csv')]

# Read all csv files as pandas dataframes and store them in a list
dfs = []
numLines = 0
for file in csv_files:
    df = pd.read_csv(file, dtype={"imdbId": str})
    # Drop extra column added while scraping by mistake
    df = df.drop(columns=['imdbid'])
    print(f"Read {file} with shape {df.shape}")
    numLines += df.shape[0]
    dfs.append(df)

# Merge all dataframes into one
merged_df = pd.concat(dfs)
print(f"Total number of lines: {numLines}", f"Shape of merged dataframe: {merged_df.shape}")

# Save merged dataframe as csv
merged_df.to_csv('box_office_collections.csv', index=False)
print("Saved merged dataframe as box_office_collections.csv")

# Get all missing_ids_*.csv files and merge them into one
missing_ids_files = [file for file in os.listdir() if file.startswith('missing_ids_') and file.endswith('.csv')]
missing_ids = []

for file in missing_ids_files:
    df = pd.read_csv(file, header=None, dtype=str)
    missing_ids.append(df)

missing_ids = pd.concat(missing_ids)
missing_ids.to_csv('missing_ids.csv', index=False, header=False)
print("Saved merged missing_ids as missing_ids.csv")