import requests
from bs4 import BeautifulSoup
import pandas as pd
import requests
from bs4 import BeautifulSoup
import pandas as pd
from requests.exceptions import ConnectionError, Timeout
from urllib3.exceptions import MaxRetryError
from urllib3.exceptions import ProtocolError
import time

# change this file location to the location of your links csv file
links_df = pd.read_csv("../Dataset/The Movies Dataset/links.csv", dtype={"imdbId": str})[3000:]
csv_stid = 0

movie_data = {}
missing_ids = []
# Counter to keep track of completed rows
completed_rows = 0

# Batch size - number of rows to scrape before saving to the CSV
batch_size = 1000  # Adjust as needed

first_batch = True

# iterate over each imdb id
for _, row in links_df.iterrows():
    imdbid = row["imdbId"]
    url = f"https://www.boxofficemojo.com/title/tt{imdbid}/"  

    response = requests.get(url, timeout=5)
    # send request to the web, request crashes after sometime so adding delays to help it recover 
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, "html.parser")
        movie_name = soup.find("h1", class_="a-size-extra-large").text.strip()
        # skip the first table giving an overview of the box office collection
        table = soup.find_all("table", {"class", "a-bordered a-horizontal-stripes a-size-base-plus"})[1:]

        movie_country_data = {}

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
                    movie_country_data[country] = collection

        # Add the movie's data to the main dictionary
        if len(movie_country_data) > 0:
            movie_data[movie_name] = movie_country_data

        completed_rows += 1

        if completed_rows % batch_size == 0:
            # Save the data to the CSV file at the end of each batch
            df = pd.DataFrame.from_dict(movie_data, orient="index")
            df.index.name = "Movie Name"
            df.reset_index(inplace=True)

            # Append the data to the CSV file
            df.to_csv(f"box_office_collection_{completed_rows/batch_size + csv_stid}.csv", mode="a", header=True, index=False, columns=df.columns)

            # Clear the movie_data dictionary for the next batch
            movie_data = {}

            print("rows completed: ", completed_rows)
            time.sleep(2)

    else:
        missing_ids.append(imdbid)
        print(imdbid)
        if len(missing_ids) % batch_size == 0:
            df_missing = pd.DataFrame.from_records(missing_ids, orient="index")
            df_missing.index.name = "Movie Name"
            df_missing.reset_index(inplace=True)

            # Append the data to the CSV file
            df_missing.to_csv("missing_ids.csv", mode="a", header=False, index=False)

            # Clear the movie_data dictionary for the next batch
            missing_ids = []
        


if movie_data:
    df = pd.DataFrame.from_dict(movie_data, orient="index")
    df.index.name = "Movie Name"
    df.reset_index(inplace=True)

    # Append the data to the CSV file
    df.to_csv("box_office_collection.csv", mode="a", header=True, index=False)

if len(missing_ids) > 0:
    df_missing = pd.DataFrame.from_dict(missing_ids, orient="index")
    df_missing.index.name = "Movie Name"
    df_missing.reset_index(inplace=True)

    # Append the data to the CSV file
    df_missing.to_csv("missing_ids.csv", mode="a", header=False, index=False)

    # Clear the movie_data dictionary for the next batch
    missing_ids = []

