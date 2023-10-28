import requests
from bs4 import BeautifulSoup
import pandas as pd



url = "https://www.boxofficemojo.com/title/tt0120338/"



movie_data = {}

# send request to the web
response = requests.get(url)

if response.status_code == 200:
    soup = BeautifulSoup(response.text, "html.parser")
    movie_name = soup.find("h1", class_="a-size-extra-large").text.strip()
    # skip the first table giving an overview of the box office collection
    table = soup.find_all("table", {"class", "a-bordered a-horizontal-stripes a-size-base-plus"})[1:]

    movie_country_data = {}

    for region in table:
        for row in region.find_all("tr")[1:]:  # Skip the header row
                cells = row.find_all("td")
                country = cells[0].text.strip()
                collection = cells[2].text.strip()
                movie_country_data[country] = collection

    # Add the movie's data to the main dictionary
    movie_data[movie_name] = movie_country_data
    
    print(soup)
else:
    print("Failed to retrieve the webpage. Status code:", response.status_code)


df = pd.DataFrame.from_dict(movie_data, orient="index")
df.index.name = "Movie Name"
df.reset_index(inplace=True)

# Save the data to a CSV file
df.to_csv("box_office_collection.csv", index=False)

