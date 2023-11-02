import pandas as pd
from collections import defaultdict

# Read bo_collections.csv file
box_office_dtypes = defaultdict(lambda: str)
box_office_dtypes['imdbId'] = str
bo_collections = pd.read_csv('../dataset/created/box_office_collections.csv', dtype=box_office_dtypes)

# Read links.csv file
links = pd.read_csv('../dataset/downloaded/links.csv', usecols=['movieId', 'imdbId'], dtype={'movieId': str, 'imdbId': str})

# Merge bo_collections and links tables on imdbId column
bo_collections = pd.merge(bo_collections, links, on='imdbId')

# Rename movieId column to id
bo_collections = bo_collections.rename(columns={'movieId': 'id'})
cols = bo_collections.columns.tolist()
cols = cols[-1:] + cols[:-1]
bo_collections = bo_collections[cols]

# Save the merged dataframe as a new csv file
bo_collections.to_csv('box_office_collections.csv', index=False)
