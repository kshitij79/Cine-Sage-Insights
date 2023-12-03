# CineSage Insights - A Predictive and Visualization Tool for Filmmakers
## CSE 6242 A Team 20

CineSage Insights is an innovative tool designed to revolutionize movie production planning through the integration of data analysis and visualization. Aimed at assisting filmmakers, it leverages a data-driven approach to influence decisions in various production stages. The tool considers key factors such as plot premise, budget, cast, release schedule, and target markets, providing predictions and visualizations regarding a film's potential success and revenue.

### Description
CineSage Insights combines advanced analytics with intuitive visualizations to guide filmmakers in their decision-making process. By analyzing data on previous movie performances, current market trends, and audience preferences, this tool offers valuable insights into the most profitable genres, optimal budget allocations, star cast impact, and suitable release windows.

### Installation
1. Clone the repository: `git clone https://github.com/CineSageInsights/repository.git`
2. Navigate to the project directory: `cd CineSageInsights`
3. Install required dependencies: `pip install -r requirements.txt`

### Execution
To explore CineSage Insights:
1. Run the Flask app: `python Flask/app.py`
2. Open a web browser and go to `http://localhost:5000/` to access the tool.
3. Explore the various features like revenue prediction, and genre-cluster analysis through the web interface.

### Live Demo
https://cinesageinsights.azurewebsites.net/

### Project Structure
The repository is organized as follows:

- `.github/workflows/main.yml`: Contains the workflow configuration for GitHub actions.
- `.gitignore`: Specifies intentionally untracked files to ignore.
- `Cluster Analysis/`: This directory includes notebooks and models for cluster analysis.
    - `ClusterUserGenre.ipynb`: Jupyter notebook for user-genre cluster analysis.
    - `gmm_model.joblib`: Saved Gaussian Mixture Model.
    - `kmeans_model.joblib`: Saved K-Means Model.
- `dataset/`: Contains various datasets used in the project.
    - `cleaned_dataset/`: Cleaned datasets for training and testing.
    - `created/`: Datasets created during the project.
    - `dataset_download_links.json`: JSON file with links to download datasets.
    - `downloaded/`: Initially downloaded datasets.
    - `download_large_csv.py`: Script to download large CSV files.
    - `README.md`: Information about the datasets.
- `Distributed CSVs/`: Contains distributed CSV files for box office collection and missing IDs.
- `Flask/`: Flask application for the web-based interface of the tool.
    - `app.py`: Main Flask application file.
    - `best_model_overall_revenue.pth`: Saved best model for revenue prediction.
    - `genre_prediction.py`: Script for genre prediction.
    - `lib/`: Libraries used in the Flask application.
    - `static/`: Static files like JavaScript and images for the Flask app.
    - `templates/`: HTML templates for the web interface.
    - `utils.py`: Utility functions for the Flask app.
- `model_experiments/`: Contains notebooks and data for various model experiments.
- `report/`: Includes report files and plots.
- `scripts/`: Scripts used for data scraping and processing.
- `README.md`: General information and overview of the repository.

### Contribution
This project is a collaborative effort of Team 20 in CSE 6242.

---

CineSage Insights is not just a tool; it's a pathway for filmmakers to embrace the digital transformation in movie production, ensuring data-driven success in the competitive world of cinema.
