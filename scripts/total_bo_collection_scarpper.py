import cpi
import requests
from bs4 import BeautifulSoup
import pandas as pd
import sys
from tqdm import tqdm
from datetime import datetime


def scrape_box_office_mojo(imdbid):
    bo_mojo_url = f"https://www.boxofficemojo.com/title/{imdbid}/"

    response = requests.get(bo_mojo_url, timeout=5)
    reveune = 0
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, "html.parser")
        worldwide = soup.find_all("div", {"class", "a-section a-spacing-none mojo-performance-summary-table"})[0]
        if worldwide: 
         revenue = worldwide.find("span", class_="money").get_text()
        else:
           return 0
    
    return int(revenue.replace('$', ''))

def update_revenue():

    movies_metadata_link = "../dataset/The Movies Dataset/movies_metadata.csv"
    movies_metadata_csv = pd.read_csv(movies_metadata_link)[0:5000]

    for idx, row in tqdm(movies_metadata_csv.iterrows(), total=movies_metadata_csv.shape[0]):
        revenue = row["revenue"]

        if revenue == 0:
            imdbid = row["imdb_id"]
            revenue = scrape_box_office_mojo(imdbid)
            

        year = datetime.strptime(row["release_date"], "%Y-%m-%d").year
        updated_revenue = cpi.inflate(revenue, year)
        movies_metadata_csv.at[idx, "revenue"] = updated_revenue

    movies_metadata_csv.to_csv(movies_metadata_link, index=False)

    

if __name__ == "__main__":
    update_revenue()