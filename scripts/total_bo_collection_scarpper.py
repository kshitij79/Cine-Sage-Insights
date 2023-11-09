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
    revenue = 0
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, "html.parser")
        worldwide = soup.find_all("div", {"class", "a-section a-spacing-none mojo-performance-summary-table"})[0]
        if worldwide: 
            revenue = worldwide.find_all("span", class_="money")

            if revenue:
                revenue = revenue[-1].get_text()
            else:
                return 0
        else:
            return 0
    else:
        return 0
    
    return float(revenue.replace('$', '').replace(',', ''))

def update_revenue():
    st_idx = 10000
    end_idx = 20000
    movies_metadata_link = "dataset/downloaded/movies_metadata.csv"
    movies_metadata_csv = pd.read_csv(movies_metadata_link)

    batch_data = movies_metadata_csv[st_idx:end_idx]

    for idx, row in tqdm(batch_data.iterrows(), total=batch_data.shape[0]):
        revenue = row["revenue"]

        if revenue == 0:
            imdbid = row["imdb_id"]
            revenue = scrape_box_office_mojo(imdbid)
            

        year = datetime.strptime(row["release_date"], "%Y-%m-%d").year
        updated_revenue = 0
        if revenue != 0:
            try:
                updated_revenue = cpi.inflate(revenue, year)
            except:
                updated_revenue = revenue 
        batch_data.at[idx, "revenue"] = updated_revenue

    movies_metadata_csv.iloc[st_idx:end_idx, :] = batch_data
    movies_metadata_csv.to_csv(movies_metadata_link, index=False)

    

if __name__ == "__main__":
    update_revenue()