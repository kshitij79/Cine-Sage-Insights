import requests
from bs4 import BeautifulSoup
import pandas as pd
import sys
from tqdm import tqdm

# Usage: python bo_collection_scrapper.py <path to links.csv> <range start (inclusive)> <range end (exclusive)>
# Example usage: python bo_collection_scrapper.py ../TheMoviesDataset/links.csv 0 12000

# Read links csv file path, range start and range end (exclusive) from input
links_csv_path = sys.argv[1]
range_start = int(sys.argv[2])
range_end = int(sys.argv[3])
links_df = pd.read_csv(links_csv_path, dtype={"imdbId": str})[range_start:range_end]

def save_data(start_row, end_row, revenue_data, missing_ids):
    print("Saving data from rows {} to {}".format(start_row, end_row))
    df = pd.DataFrame(revenue_data).set_index("imdbId")
    df_columns = ["imdbId", "Movie Name"] + sorted([x for x in df.columns if x not in ("imdbId", "Movie Name")])
    df = df.reindex(df_columns, axis=1)
    df.to_csv(f"box_office_collection_{start_row}_{end_row}.csv", header=True)
    
    missing_df = pd.Series(missing_ids)
    missing_df.to_csv(f"missing_ids_{start_row}_{end_row}.csv", index = False, header = False)


revenue_data = []
missing_ids = []

# Batch size - number of rows from links.csv to scrape before saving to the CSV
batch_size = 1000  # Adjust as needed

row_number = range_start

for _, row in tqdm(links_df.iterrows(), total=links_df.shape[0]):
    imdbid = row["imdbId"]
    url = f"https://www.boxofficemojo.com/title/tt{imdbid}/"

    response = requests.get(url, timeout=5)
    # send request to the web, request crashes after sometime so adding delays to help it recover 
    if response.status_code == 200:
        movie_row = { "imdbId": imdbid }
        soup = BeautifulSoup(response.text, "html.parser")
        movie_row["Movie Name"] = soup.find("h1", class_="a-size-extra-large").text.strip()
        # skip the first table giving an overview of the box office collection
        table = soup.find_all("table", {"class", "a-bordered a-horizontal-stripes a-size-base-plus"})[1:]

        for region in table:
            column = 0
            rows = region.find_all("tr")
            cells = rows[0].find_all("th")

            for i, cell in enumerate(cells):
                if cell.text.strip() in {"Lifetime Gross", "Gross"}:
                    column = i
                    break

            if column == 0:
                continue

            for row in rows[1:]:  # Skip the header row
                    cells = row.find_all("td")
                    country = cells[0].text.strip()
                    collection = cells[column].text.strip()
                    movie_row[country] = collection

        # Add the movie's data to the main dictionary
        if len(movie_row) > 2:
            revenue_data.append(movie_row)
        else:
            missing_ids.append(imdbid)

    else:
        missing_ids.append(imdbid)
        print(imdbid, "missing")
    
    row_number += 1
    
    # If completed a batch, save the data
    if row_number % batch_size == 0:
        save_data(row_number-batch_size, row_number, revenue_data, missing_ids)
        revenue_data = []
        missing_ids = []
        
if row_number % batch_size != 0:
    save_data(row_number-row_number%batch_size, row_number, revenue_data, missing_ids)